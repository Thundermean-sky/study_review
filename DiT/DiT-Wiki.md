# 简化版 DiT 实现 Wiki

> 目标：手把手从零实现一个简化版 DiT（Diffusion Transformer），**跑通完整训练流程**。
> 简化策略：减少 Transformer Block 数量、降低隐层维度；其余细节（latent 空间、patch 化、AdaLN 调制、zero-init、DDPM 训练目标）完全还原原版 DiT 论文与官方实现。
> 本 wiki 面向"理解优先"的学习者，记录每一步的设计意图与踩坑。

---

## 1. 项目概览

### 1.1 什么是 DiT

DiT（《Scalable Diffusion Models with Transformers》）的核心思想：**把扩散模型的去噪网络从 U-Net 换成 Transformer**。扩散模型在"latent 空间"（由 VAE 编码得到）逐步加噪 / 去噪，DiT 负责在 latent 上预测噪声。

### 1.2 简化对照表

| 项目 | 原版 DiT-S | 本实现 | 说明 |
|---|---|---|---|
| Block 数 | 12 | 2 | 主要简化点 |
| 隐层维度 | 384 | 256 | 次要简化点 |
| 参数量 | ~33M | ~2.5M | |

### 1.3 完全还原的细节

- VAE latent 4 通道、空间下采样 8 倍
- patch 大小 2×2，token 数 = 16×16 = 256
- sinusoidal 时间嵌入（frequency_embedding=256）
- **AdaLN-zero** 调制（DiT 核心创新）
- MLP ratio = 4，SiLU / GELU(tanh) 激活
- 类别嵌入 + label dropout（10%，为 classifier-free guidance 做准备）
- FinalLayer 与 adaLN 调制层 **zero-init**
- 模型不设位置编码（原版消融后移除）

---

## 2. 架构总览

```
图片 x (B, 3, 256, 256)
   │
   ▼ 模拟 VAE Encoder（卷积下采样 ×8）
latent z (B, 4, 32, 32)              ← latent 空间
   │
   ▼ Patchify + PatchEmbed（Conv2d kernel=2, stride=2）
tokens (B, 256, 256)                 ← 256 个 token，每个 256 维
   │
   ▼ 2 × DiTBlock（AdaLN 调制 + 多头自注意力 + MLP）
tokens (B, 256, 256)
   │                                ← 条件 c = 时间嵌入 + 类别嵌入
   ▼ FinalLayer（LayerNorm + Linear, zero-init）
tokens (B, 256, 16)                  ← 投影回 patch 维度
   │
   ▼ Unpatchify
预测噪声 ε_θ (B, 4, 32, 32)          ← 训练损失在 latent 空间计算
```

关键点：**去噪在 latent 空间进行**。训练时输入是"加噪后的 latent"（不是图片），损失是 `MSE(ε_θ(z_t, t, y), ε)`。

---

## 3. 文件结构与运行

```
DiT/
├── config.py        # 所有超参数
├── components.py    # 基础组件：嵌入、Patch、注意力、DiTBlock
├── vae.py           # 模拟 VAE：Encoder / Decoder
├── model.py         # FinalLayer + DiT 主模型
└── train.py         # DDPM 训练循环（假数据）
```

运行：`python train.py`（预期 loss 从 ~1 开始，假数据下基本维持 ~1，详见 §12.5）

---

## 4. 配置（config.py）

```python
IMAGE_SIZE = 256            # 输入图片边长（像素）
IN_CHANNELS = 3             # RGB
NUM_CLASSES = 10            # 类别数

LATENT_CHANNELS = 4         # 真实 VAE latent 通道数就是 4
LATENT_SIZE = IMAGE_SIZE // 8   # = 32，模拟 VAE 下采样 8 倍
LATENT_SHAPE = (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)

PATCH_SIZE = 2
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * LATENT_CHANNELS   # = 16
NUM_PATCHES = (LATENT_SIZE // PATCH_SIZE) ** 2          # = 256

HIDDEN_SIZE = 256
DEPTH = 2
NUM_HEADS = 8
MLP_RATIO = 4
FREQ_EMB_SIZE = 256

NUM_TIMESTEPS = 1000
```

尺寸链路：`256 → 32 → 16×16=256 tokens → 每个 patch 2×2×4=16 维`。

---

## 5. 模块一：时间步与类别嵌入

### 5.1 为什么需要时间步嵌入

扩散过程分步加噪 / 去噪，t=0 是干净 latent，t=T 是纯噪声。模型在 t 不同时行为不同（早期去轮廓、后期去噪点），必须把"当前第几步"告诉网络。用 sinusoidal 嵌入：无额外参数、平滑连续、编码不同尺度。

### 5.2 sinusoidal 公式

```
对第 i 维（i 从 0 到 half-1）：
    emb[2i]   = cos(t / 10000^(2i/dim))
    emb[2i+1] = sin(t / 10000^(2i/dim))
```

代码用 `10000^(-i/half) = exp(-i/half·ln(10000))` 一次性算出频率向量再广播：

```python
def timestep_embedding(t, dim=FREQ_EMB_SIZE, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]        # 外积: (B, 1) × (1, half)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # cos 在前（官方顺序）
    return embedding
```

### 5.3 TimeEmbedder / LabelEmbedder

```python
class TimeEmbed(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, frequency_embedding_size=FREQ_EMB_SIZE):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),                                   # 原版用 SiLU
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        return self.mlp(timestep_embedding(t))           # (B, 256)


class LabelEmbedding(nn.Module):
    def __init__(self, num_class=NUM_CLASSES, hidden_size=HIDDEN_SIZE, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_class + 1, hidden_size)   # 多一个"无条件"槽位
        self.num_classes = num_class
        self.dropout_prob = dropout_prob

    def token_drop(self, labels):
        drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train=True):
        if train and self.dropout_prob > 0:
            labels = self.token_drop(labels)             # 10% 概率丢成无条件
        return self.embedding(labels)
```

- label dropout 为 classifier-free guidance 采样做准备（训练时学会无条件生成）
- **条件合并**：`c = t_emb + y_emb`，直接相加后传给每个 Block

---

## 6. 模块二：模拟 VAE（卷积实现）

### 6.1 设计原则

VAE 只是"特征提取器"：图片 ↔ latent 的转换。DiT 不关心转换细节，只要求**输出维度与真实 VAE 一致**。因此用几个卷积假装即可，不需要 KL 重参数化，也不需要训练。

### 6.2 Encoder

3 次 `kernel=3, stride=2, padding=1` 卷积：尺寸 256→128→64→32（下采样 8 倍），通道 3→64→128→256，最后 `stride=1` 卷积投影到 4 通道：

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 64, kernel_size=3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.SiLU(),
        )
        self.to_latent = nn.Conv2d(256, LATENT_CHANNELS, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.to_latent(self.down(x))   # (B,3,256,256) → (B,4,32,32)
```

### 6.3 Decoder

`Upsample(bilinear) + 卷积` 上采样 3 次（32→64→128→256），通道 4→256→128→64→3。训练流程不用 Decoder（训练吃 latent），只有采样生成图片时才会用到。

---

## 7. 模块三：Patchify / PatchEmbed / Unpatchify

### 7.1 核心洞察：`Conv2d(kernel=2, stride=2)` 就是 patch 化

- kernel = patch 大小，stride = patch 大小 → 输出每个位置对应输入一个**互不重叠**的 2×2 邻域
- 每个输出通道 = 对 2×2×4=16 个输入值做加权求和
- 一次完成"展平 + 线性投影"：16 维 patch → 256 维 token

```python
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=LATENT_CHANNELS, patch_size=PATCH_SIZE, embed_dim=HIDDEN_SIZE):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)          # (B,4,32,32) → (B,256,16,16)
        x = x.flatten(2)          # (B,256,256) 通道维在前
        x = x.transpose(1, 2)     # (B,256,256) 换成 token 维在前（Transformer 约定）
        return x
```

### 7.2 Unpatchify（逆操作）

```python
def unpatchify(x, patch_size=PATCH_SIZE, in_channels=LATENT_CHANNELS):
    h = w = int(x.shape[1] ** 0.5)                          # 256 token → 16×16
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, in_channels)  # (B,16,16,2,2,4)
    x = torch.einsum('nhwpqc -> nchwpq', x)                 # 交换维序
    return x.reshape(x.shape[0], in_channels, patch_size * h, patch_size * w)  # (B,4,32,32)
```

### 7.3 重要概念（易错点）

**`unpatchify` 不是 `PatchEmbed` 的逆操作**。PatchEmbed 把 16 维投影到 256 维，信息不可无损还原。正确调用链是：`FinalLayer 把 256 维投影回 16 维 → 再 unpatchify`。unpatchify 的输入必须是 `(B, 256, 16)`。

### 7.4 位置编码

**DiT 官方没有位置编码**。论文消融实验（sinusoidal / learned / 无）差异极小，最终版本移除。原因：patch 数量少、顺序固定、latent 保留空间结构。

---

## 8. 模块四：多头自注意力（MHSA）

### 8.1 原理

每个 token 与其他所有 token 计算相关性并加权汇总：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

- `√d_k` 缩放：Q·K 的方差随 d_k 增大，softmax 会趋于 one-hot、梯度消失，除以 √d_k 归一化方差
- 多头：切到不同子空间学习不同关系（局部纹理 / 全局结构）

### 8.2 维度变化

```
(B, 256, 256) → Linear(256→768) → reshape(B,256,3,8,32) → permute → (3,B,8,256,32)
→ Q,K,V 各 (B,8,256,32) → 注意力 (B,8,256,256) → 加权 → (B,8,256,32)
→ transpose(1,2).reshape(B,256,256) → 输出投影 Linear
```

### 8.3 代码

```python
class Attention(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, heads=NUM_HEADS):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)   # 一个 Linear 生成 QKV
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)  # 先按内存顺序切
        qkv = qkv.permute(2, 0, 3, 1, 4)                      # QKV 轴挪到最前
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)            # 合并 8 个头
        return self.proj(out)
```

### 8.4 核心坑：QKV 拆分顺序（见 §12.1）

**必须** `reshape(B, N, 3, heads, head_dim)` 再 `permute`。若直接 `reshape(3, B, heads, N, head_dim)`，reshape 不做数据搬运，会把 Q/K/V 三段和 token 顺序全部打乱——**shape 不报错、loss 能下降，但语义完全错误**。

---

## 9. 模块五：AdaLN 调制（DiT 核心创新）

### 9.1 为什么不用普通 LayerNorm

普通 LN 的 γ、β 是固定参数，对所有 token、所有样本都一样。但扩散模型每个样本的条件（时刻 t、类别 y）都不同，固定参数无法区分。

### 9.2 AdaLN：条件预测调制量

条件向量 c 通过一个小网络预测 6 个量（注意力 + MLP 两个分支，各含 shift / scale / gate）：

```python
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO):
        super().__init__()
        self.norm_1 = nn.LayerNorm(hidden_size, elementwise_affine=False)  # 仿射参数由调制量扮演
        self.norm_2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads)
        mlp_hidden = hidden_size * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )
        # zero-init：初始时调制量为 0，block 退化为恒等映射
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm_1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_2(x), shift_mlp, scale_mlp))
        return x
```

### 9.3 三个设计细节（重点理解）

1. **`elementwise_affine=False`**：LN 的静态 γ、β 与动态 scale/shift 是重复自由度，关掉避免冗余参数
2. **`1 + scale`**：LN 输出已归一化（均值 0、方差 1），若直接 `x*scale` 则初始化时全部输出 0，信息被抹掉；用 `1+scale` 从"不调制"（恒等）出发，scale 可双向平滑调整
3. **zero-init + gate**：gate=0 使分支初始不贡献 → 整个 block 初始等于恒等映射 → 网络从"浅层等价"开始学，训练稳定

---

## 10. 模块六：最终层 + 完整模型

### 10.1 FinalLayer

```python
class FinalLayer(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, patch_size=PATCH_SIZE, out_channels=LATENT_CHANNELS):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        nn.init.constant_(self.linear.weight, 0)   # zero-init：初始输出全 0
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(self.norm_final(x))   # (B,256,256) → (B,256,16)
```

zero-init 意义：模型初始"预测噪声 = 0"，loss ≈ 1（噪声能量），从稳定起点开始学。

### 10.2 完整 DiT

```python
class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_embed = TimeEmbed()
        self.y_embed = LabelEmbedding()
        self.patch_embed = PatchEmbed()
        self.blocks = nn.ModuleList([DiTBlock() for _ in range(DEPTH)])
        self.final_layer = FinalLayer()

    def forward(self, x, t, y):
        x = self.patch_embed(x)                    # (B,256,256)
        t = self.t_embed(t)                        # (B,256)
        y = self.y_embed(y, self.training)         # (B,256)
        c = t + y                                  # 条件合并
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x)                    # (B,256,16)
        return unpatchify(x)                       # (B,4,32,32)
```

注意：
- 输入是 **latent**（`LATENT_SHAPE`），不是原始图片
- `y_embed(y, self.training)`：训练时开 label dropout，推理时关闭

---

## 11. 模块七：DDPM 训练循环

### 11.1 训练目标

```
加噪: z_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε，其中 ᾱ_t = ∏(1-β_i)（预先算好，非参数）
模型: ε_θ(z_t, t, y) → 预测噪声
损失: MSE(ε_θ(z_t, t, y), ε)
```

### 11.2 代码

```python
betas = torch.linspace(1e-4, 0.02, NUM_TIMESTEPS)      # 线性 beta schedule
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)          # ᾱ_t

def q_sample(x0, t, noise):
    alphas_bar = alphas_cumprod.to(device=t.device)[t]  # 索引张量需与 t 同设备！
    sqrt_alpha_bar = torch.sqrt(alphas_bar)[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1.0 - alphas_bar)[:, None, None, None]
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

def make_fake_data(batch_size, encoder, device):
    img = torch.randn(batch_size, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():                              # encoder 不训练
        return encoder(img)

def train(epochs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    model = DiT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        batch_size = 4
        x0 = make_fake_data(batch_size, encoder, device)
        t = torch.randint(0, NUM_TIMESTEPS, (batch_size,), device=device)
        noise = torch.randn_like(x0)                   # 必须是标准正态！
        z_t = q_sample(x0, t, noise)
        y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        pred = model(z_t, t, y)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
```

---

## 12. 踩坑记录（复习重点）

### 12.1 QKV 拆分顺序（最隐蔽的语义 bug）

`qkv(x)` 输出 `(B, N, 768)`，内存顺序是"每 token 内 Q 段(256) → K 段(256) → V 段(256)"。
- 正确：`reshape(B, N, 3, heads, head_dim)` → `permute(2, 0, 3, 1, 4)`，QKV 三段干净分开
- 错误：直接 `reshape(3, B, heads, N, head_dim)`，reshape 不搬数据，按行优先重新解释 → Q 里混入 K/V，token 顺序也被打乱

**规则**：`reshape` 只按内存顺序"切块"，`permute` 才负责"挪轴"。QKV 三段在内存中紧挨着，所以"3"轴必须紧跟特征维（`(B, N, 3, ...)`），再用 permute 提前。

**排查手段**：语义自检 `q == F.linear(x, qkv_w[:256], qkv_bias[:256])`，`allclose` 应为 True。

### 12.2 测试代码漏 bias

`nn.Linear` 默认 `bias=True`。`F.linear(x, w[:256])` 忘了传 `bias[:256]`，与真实 Q 段差一个 bias → `allclose=False`。**写自检代码时不要把 bias 漏掉**。

### 12.3 `rand` vs `randn`

- `torch.rand`：均匀分布 [0,1)，**不是**噪声的正确分布
- DDPM 的噪声 ε 必须是标准正态 N(0,1)：`torch.randn_like(x0)`

这种错误"能跑、loss 也下降"，但学的语义全错，属于最需警惕的一类。

### 12.4 设备不匹配

- `alphas_cumprod`（模块级 CPU 张量）被 GPU 的 `t` 索引 → `RuntimeError: indices should be on cpu or same device`。修复：`alphas_cumprod.to(device=t.device)[t]`
- `y` 没指定 device 会留在 CPU，查表时报错。**所有参与前向的张量都要在同一设备**

### 12.5 假数据 loss 卡在 ~1（不是 bug！）

现象：每步生成全新随机图片 → latent，训练 1000 步 loss 始终 ~1.0。

原因：无规律随机数据里，模型能学到的唯一稳定策略是"输出 0"（`E[ε²]=1` 是最优解），没有更多可学信息。loss 卡 1 **恰好证明训练链路是通的**（模型学到了能学的一切）。

**诊断方法**（区分代码 bug 与数据问题）：
1. 固定一个 batch，反复训练过拟合 → 若 loss 能降到 ~0，说明模型与梯度链路正常
2. 检查参数是否变化：`(p_after - p_before).abs().sum()`

想看到 loss 下降，可用**固定小数据集**（预生成 8 张图，每步从中抽样），让 latent 具备可学统计结构；或加大 lr 与步数。

---

## 13. 验证与自检清单

| 模块 | 检查点 | 期望 |
|---|---|---|
| 嵌入 | `TimeEmbed`(t) / `LabelEmbedding`(y) | `(2, 256)`，相加后 `(2, 256)` |
| VAE | `Encoder`(img) | `(2, 4, 32, 32)` |
| VAE | `Decoder`(z) | `(2, 3, 256, 256)` |
| Patch | `PatchEmbed`(latent) | `(2, 256, 256)` |
| Patch | `unpatchify((2,256,16))` | `(2, 4, 32, 32)` |
| Attention | 输入输出 shape 不变 | `(2, 256, 256)` |
| Attention | Q 段自检 `allclose` | `True` |
| DiTBlock | 初始化时（zero-init）恒等 | `allclose(out, x)` 为 True |
| DiT | 完整前向 | `(2, 4, 32, 32)`，参数 ~2.5M |
| 训练 | 假数据跑通 | 无报错，loss 从 ~1 开始 |

---

## 14. 后续扩展方向

1. **采样生成（DDPM sampling）**：从纯噪声 `z_T ~ N(0,1)` 开始，按 `z_{t-1} = (z_t - β_t/√(1-ᾱ_t)·ε_θ)/√(1-β_t) + σ_t·noise` 迭代去噪，最后用 Decoder 还原图片
2. **Classifier-free guidance**：推理时 `ε_θ = w·ε_θ(cond) + (1-w)·ε_θ(uncond)`，配合训练时已有的 label dropout
3. **真正的 VAE**：换用 Stable Diffusion 的 SD-VAE 编码器，替代模拟卷积
4. **缩放规律**：探索 depth / width / heads 对训练的影响（DiT 论文的核心实验）
5. **EMA、学习率调度（warmup + cosine）**、混合精度等训练工程细节
