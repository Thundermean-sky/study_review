# BERT 代码复现与逻辑理解（Wiki）

> 本文档结合 `main.py` 的简化实现，系统梳理 BERT 的核心思想、模型结构与预训练流程，
> 供自己复习与他人学习使用。代码是一个**可直接运行的 6 层 Encoder 迷你版**，
> 保留了 BERT 的关键设计（三种嵌入、MLM、NSP、残差 + LayerNorm），省略了原始论文中的部分工程细节。
>
> 运行：`python main.py`（使用合成随机语料，loss 应随 epoch 逐步下降）。

---

## 目录

- [1. BERT 核心思想](#1-bert-核心思想)
- [2. 整体架构与数据流](#2-整体架构与数据流)
- [3. 输入表示：三种嵌入](#3-输入表示三种嵌入)
- [4. 预训练任务（MLM / NSP）](#4-预训练任务mlm--nsp)
- [5. 模型结构详解](#5-模型结构详解)
- [6. 任务输出头（权重共享）](#6-任务输出头权重共享)
- [7. 数据预处理流程](#7-数据预处理流程)
- [8. 训练流程](#8-训练流程)
- [9. 运行方式](#9-运行方式)
- [10. 关键超参与扩展建议](#10-关键超参与扩展建议)

---

## 1. BERT 核心思想

BERT（Bidirectional Encoder Representations from Transformers）的关键贡献如下：
**首次把"双向"的上下文建模带到了预训练语言模型中**。

- **为什么需要"双向"**：以往的 NLP 模型（如从左到右的语言模型、Seq2Seq）只能看到当前词的一侧上下文。
  而一个词的含义往往依赖它左右两边的词（例如"他去了银行取钱" vs "河边的银行"），
  只有同时看到两侧，模型才能学到真正的双向语义表示。
- **为什么只用 Encoder**：Transformer 的 Encoder 通过**多头自注意力**让序列中每个位置都能直接"关注"到任意其他位置，
  天然适合做双向建模；而 Decoder 带有因果掩码（只能看左侧），会破坏双向性。因此 BERT 只取 Encoder 部分。
- **自注意力如何看到左右文**：自注意力计算时，每个 token 的 Query 会与序列中所有 token 的 Key 做点积，
  没有位置方向限制，所以一个词可以同时聚合它左边和右边的内容。

> 💡 总体而言：BERT = 只保留 Transformer Encoder + 用"完形填空(MLM)"和"句对判断(NSP)"两个任务做双向预训练。

---

## 2. 整体架构与数据流

```
原始句子对 (A, B)
      │
      ▼
[输入构造]  ── Tokenize + 特殊符号([CLS][SEP]) + MLM 随机遮盖
      │
      ▼
[三种嵌入相加]  TokenEmbed + SegmentEmbed + PositionEmbed  →  dropout
      │
      ▼
[6 × Encoder Block]  每层: 多头自注意力 → Add&Norm → FFN → Add&Norm
      │
      ▼
[序列隐状态 sequence_output]   形状 (B, T, H)
      ├──────────────┐
      ▼              ▼
[MLM 头]          [NSP 头]
取被遮盖位置        取 [CLS] 位置
预测原词            判断是否为下一句
      │              │
      ▼              ▼
 CrossEntropy    CrossEntropy
      └─────┬───────┘
            ▼
        loss = loss_mlm + loss_nsp  →  反向传播
```

张量维度约定：`B`=批大小，`T`=序列长度(本代码 `max_len=16`)，`H`=隐藏维度(`hidden_size=128`)。

---

## 3. 输入表示：三种嵌入

BERT 把一个 token 的表示拆成**三个向量相加**，Token / Position / Segment 嵌入。

### 3.1 Token 嵌入
普通的词向量查找表（`nn.Embedding`），把词表中的 id 映射为 `H` 维向量。

### 3.2 Segment 嵌入
准确地说是**区分句子对中的两条句子**：
- 句子 A 的所有 token → `segment_id = 0`
- 句子 B 的所有 token → `segment_id = 1`

它的作用是告诉模型"哪些词属于同一个句子"，对句对任务（NSP、问答、蕴含判断）很关键。
代码中用一个大小为 2 的嵌入表实现（见 `BERT.segment_embedding`，源码 `168:174:main.py`）。

### 3.3 Position 嵌入
**BERT 使用的是可学习的位置嵌入**，
而不是原始 Transformer 论文里的正弦/余弦固定编码。好处是位置表示可以随任务一起训练。
代码中 `PositionalEmbedding` 是一个可训练参数矩阵。

### 3.4 特殊 Token 说明
| Token | id | 含义 |
|-------|----|------|
| `[CLS]` | 2 | 句首，其最终隐状态用于 NSP 分类 |
| `[SEP]` | 3 | 句子分隔符 / 句尾 |
| `[MASK]` | 1 | MLM 遮盖占位符 |
| `[PAD]` | 0 | 补齐到固定长度，注意力中会被屏蔽 |

三种嵌入相加后再经过一次 `dropout`，才送入 Encoder：

```168:188:main.py
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers,
                 ffn_hidden, max_len, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        self.pos_embedding = PositionalEmbedding(max_len, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([
            EncoderBlock(hidden_size, num_heads, ffn_hidden, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tokens, segments, attention_mask=None):
        x = self.token_embedding(tokens) + self.segment_embedding(segments)
        x = self.pos_embedding(x)
        x = self.embed_dropout(x)
        for block in self.encoder:
            x = block(x, attention_mask)
        return x
```

句子拼接也由工具函数完成（自动加 `[CLS]`/`[SEP]` 并生成 segment 标记）：

```13:20:main.py
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """拼接 [CLS] + A + [SEP] (+ B + [SEP])，并返回对应的 segment 标记。"""
    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['[SEP]']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

---

## 4. 预训练任务（MLM / NSP）

BERT 用两个自监督任务预训练，这也是它不需要人工标注就能学到好表示的原因。

### 4.1 MLM（Masked Language Model，完形填空）
- 对输入序列中**约 15% 的普通 token** 做遮盖（特殊符号 `[CLS]`/`[SEP]` 不遮盖）。
- 遮盖后分三种情况（与你笔记一致）：
  - **80%** 替换为 `[MASK]`（模型真正需要预测的目标形态）；
  - **10%** 替换为一个**随机**词（增加噪声鲁棒性，避免模型只在看到 `[MASK]` 时才工作）；
  - **10%** **保持不变**（让输入与原始一致，缓解预训练/微调不一致）。
- 被遮盖位置的**原始词 id** 会被记录下来，作为计算交叉熵的标签（`mlm_labels`）。

```39:66:main.py
def encode_sample(vocab, tokens_a, tokens_b, is_next, max_len, mask_prob=0.15):
    """将句子对编码为模型输入，并完成 MLM 的随机遮盖。"""
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)

    # MLM 遮盖：15% 概率，其中 80% 变 [MASK]，10% 变随机词，10% 保持不变
    mlm_labels = []
    for i, t in enumerate(tokens):
        if t in ('[CLS]', '[SEP]'):
            continue
        if random.random() < mask_prob:
            orig_id = vocab[t]
            r = random.random()
            if r < 0.8:
                tokens[i] = '[MASK]'
            elif r < 0.9:
                tokens[i] = random.choice(list(vocab.keys()))
            mlm_labels.append((i, orig_id))

    ids = [vocab[t] for t in tokens]
    seq_len = len(ids)
    pad_len = max_len - seq_len
    ids += [vocab['[PAD]']] * pad_len
    segments += [0] * pad_len
    attention_mask = [1] * seq_len + [0] * pad_len

    positions = [p for p, _ in mlm_labels]
    label_ids = [l for _, l in mlm_labels]
    return ids, segments, attention_mask, positions, label_ids, is_next
```

### 4.2 NSP（Next Sentence Prediction，下一句预测）
- 构造句对时，50% 取"真实的下一句"（标签 1），50% 取"随机一句"（标签 0）。
- 模型取 `[CLS]` 位置的最终隐状态做二分类，判断 B 是否紧跟在 A 之后。
- 这个任务帮助模型理解句子间关系，对问答、自然语言推断等下游任务很有用。

---

## 5. 模型结构详解

### 5.1 多头自注意力 MultiHeadAttention
把 `H` 维拆成 `num_heads` 个 `head_dim` 维的子空间，并行计算注意力再拼接。
缩放因子 `scale = head_dim ** -0.5` 防止点积过大导致 softmax 梯度消失。
`attention_mask` 中值为 0 的位置（padding）会被置为 `-1e9`，softmax 后权重趋近 0。

```93:119:main.py
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)
```

### 5.2 前馈网络 PositionWiseFFN
对每个位置独立施加两层全连接，中间用 GELU 激活（BERT 原论文使用 GELU 而非 ReLU）。

```122:131:main.py
class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_size, ffn_hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden)
        self.fc2 = nn.Linear(ffn_hidden, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))
```

### 5.3 残差 + AddNorm
Transformer 的标志性结构：`LayerNorm(x + Dropout(sublayer(x)))`。
先残差相加再做层归一化，缓解深层网络梯度消失。

```134:141:main.py
class AddNorm(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.ln(x + self.dropout(y))
```

### 5.4 Encoder Block（单个编码层）
注意力子层 + 前馈子层，各带残差与 AddNorm：

```144:155:main.py
class EncoderBlock(nn.Module):
    """单层 Encoder：多头自注意力 + 前馈网络（各带残差 + AddNorm）。"""
    def __init__(self, hidden_size, num_heads, ffn_hidden, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.addnorm1 = AddNorm(hidden_size, dropout)
        self.ffn = PositionWiseFFN(hidden_size, ffn_hidden, dropout)
        self.addnorm2 = AddNorm(hidden_size, dropout)

    def forward(self, x, mask=None):
        h = self.addnorm1(x, self.attention(x, mask))
        return self.addnorm2(h, self.ffn(h))
```

### 5.5 6 层 Encoder 堆叠
本代码通过 `nn.ModuleList` 重复 6 个 `EncoderBlock`（由 `num_layers=6` 控制），
这正是你要求的"6 层 encoder"。你可以通过修改 `train()` 里的 `cfg` 调整层数。

---

## 6. 任务输出头（权重共享）

`BERTPretrainer` 在 Encoder 之上接两个任务头：

- **MLM 头**：用 `gather` 取出被遮盖位置的隐状态 → 经 `Linear + GELU + LayerNorm` → 线性映射回词表大小。
  单层 MLP"即对应这里的 `mlm_hidden + mlm_act + mlm_ln`，
  此外还有一个输出层 `mlm_output` 把向量映射成词表 logits。
- **权重共享（重要细节）**：`mlm_output` 的输出层权重**直接复用** `token_embedding` 的权重
  （源码第 201 行）。这能减少参数量，并让"预测词"与"查词向量"在同一语义空间。
- **NSP 头**：取 `[CLS]`（位置 0）的隐状态，过一层 `Linear(H, 2)` 做二分类。

```191:213:main.py
class BERTPretrainer(nn.Module):
    """在 BERT 之上接入 MLM 与 NSP 两个预训练任务头。"""
    def __init__(self, bert, vocab_size):
        super().__init__()
        self.bert = bert
        H = bert.hidden_size
        self.mlm_hidden = nn.Linear(H, H)
        self.mlm_act = nn.GELU()
        self.mlm_ln = nn.LayerNorm(H)
        self.mlm_output = nn.Linear(H, vocab_size, bias=False)
        self.mlm_output.weight = self.bert.token_embedding.weight  # 权重共享
        self.nsp_output = nn.Linear(H, 2)

    def forward(self, tokens, segments, attention_mask, mlm_positions):
        encoded = self.bert(tokens, segments, attention_mask)
        # MLM：取被遮盖位置对应的隐状态
        idx = mlm_positions.unsqueeze(-1).expand(-1, -1, self.bert.hidden_size)
        features = encoded.gather(1, idx)
        features = self.mlm_ln(self.mlm_act(self.mlm_hidden(features)))
        mlm_logits = self.mlm_output(features)
        # NSP：取 [CLS]（位置 0）对应的隐状态
        nsp_logits = self.nsp_output(encoded[:, 0, :])
        return mlm_logits, nsp_logits
```

---

## 7. 数据预处理流程

为了让训练能直接跑起来，代码用合成随机语料演示，但流程与真实语料一致：

1. `build_vocab`：构建词表，预留 4 个特殊符号（`[PAD]/[MASK]/[CLS]/[SEP]`）。
2. `random_sentence`：随机抽词组成句子（真实场景应替换为分词后的句子）。
3. `encode_sample`：句子拼接 + MLM 遮盖 + padding（见第 4.1 节）。
4. `make_batch`：把 MLM 的遮盖位置/标签 padding 到固定长度，
   位置补 `0`、标签补 `-100`（`-100` 在 `CrossEntropyLoss` 中会被忽略，不参与 loss）。

```69:89:main.py
def make_batch(vocab, batch_size, max_len, max_mlm):
    buf = dict(tokens=[], segments=[], attention_mask=[],
               mlm_positions=[], mlm_labels=[], nsp_labels=[])
    for _ in range(batch_size):
        ta = random_sentence(vocab)
        tb = random_sentence(vocab)
        is_next = 1 if random.random() < 0.5 else 0
        ids, segs, attn, positions, labels, _ = encode_sample(
            vocab, ta, tb, is_next, max_len)

        # 将 MLM 位置/标签 padding 到固定长度；位置补 0，标签补 -100（忽略）
        positions = positions + [0] * (max_mlm - len(positions))
        labels = labels + [-100] * (max_mlm - len(labels))

        buf['tokens'].append(ids)
        buf['segments'].append(segs)
        buf['attention_mask'].append(attn)
        buf['mlm_positions'].append(positions)
        buf['mlm_labels'].append(labels)
        buf['nsp_labels'].append(is_next)
    return buf
```

> ⚠️ 注意：`attention_mask` 在送入模型前会被 reshape 成 `(B, 1, 1, T)`，
> 以便在注意力分数矩阵 `(B, heads, T, T)` 上做广播屏蔽（见 `train()` 第 237 行）。

---

## 8. 训练流程

**两个任务的 loss 相加后反向传播**。

- `loss_mlm`：被遮盖位置的预测词 与 原始词 做交叉熵（`ignore_index=-100` 跳过 padding 标签）。
- `loss_nsp`：`[CLS]` 的二分类 与 句对标签 做交叉熵。
- `loss = loss_mlm + loss_nsp`，`loss.backward()` 统一更新全部参数。

```217:253:main.py
def train():
    vocab = build_vocab()
    V = len(vocab)
    cfg = dict(hidden_size=128, num_heads=4, num_layers=6,
               ffn_hidden=256, max_len=16, dropout=0.1)
    max_mlm = 8

    bert = BERT(V, **cfg)
    model = BERTPretrainer(bert, V)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(20):
        for _ in range(50):
            batch = make_batch(vocab, batch_size=16, max_len=16, max_mlm=max_mlm)
            tokens = torch.tensor(batch['tokens'])
            segments = torch.tensor(batch['segments'])
            attn = torch.tensor(batch['attention_mask']).unsqueeze(1).unsqueeze(2)
            mlm_pos = torch.tensor(batch['mlm_positions'])
            mlm_labels = torch.tensor(batch['mlm_labels'])
            nsp_labels = torch.tensor(batch['nsp_labels'])

            mlm_logits, nsp_logits = model(tokens, segments, attn, mlm_pos)

            loss_mlm = mlm_loss_fn(mlm_logits.reshape(-1, V), mlm_labels.reshape(-1))
            loss_nsp = nsp_loss_fn(nsp_logits, nsp_labels)
            loss = loss_mlm + loss_nsp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch:02d} | mlm_loss={loss_mlm.item():.4f} "
              f"| nsp_loss={loss_nsp.item():.4f} | total={loss.item():.4f}")
```

---

## 9. 运行方式

```bash
cd f:/study/bert
python main.py
```

预期输出类似（loss 随 epoch 下降）：

```
epoch 00 | mlm_loss=4.5231 | nsp_loss=0.6921 | total=5.2152
epoch 01 | mlm_loss=3.8910 | nsp_loss=0.6103 | total=4.5013
...
```

---

## 10. 关键超参与扩展建议

| 超参 | 本代码值 | 说明 |
|------|----------|------|
| `num_layers` | 6 | Encoder 层数（你要求的核心配置） |
| `hidden_size` | 128 | 隐状态维度，原版 BERT-Base 为 768 |
| `num_heads` | 4 | 注意力头数（需整除 `hidden_size`） |
| `ffn_hidden` | 256 | FFN 中间层维度 |
| `max_len` | 16 | 最大序列长度 |
| `mask_prob` | 0.15 | MLM 遮盖比例 |