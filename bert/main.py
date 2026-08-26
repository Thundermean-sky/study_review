"""简化版 BERT 预训练代码（6 层 Encoder，含 MLM + NSP 任务）。

可直接运行：python main.py
使用合成随机语料演示完整训练流程，loss 应逐步下降。
"""
import random
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------- 工具函数 -------------------------
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """拼接 [CLS] + A + [SEP] (+ B + [SEP])，并返回对应的 segment 标记。"""
    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['[SEP]']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


# ------------------------- 词汇表与数据 -------------------------
def build_vocab():
    words = ['天气', '今天', '明天', '很好', '不好', '我们', '你们', '去', '公园', '学校',
             '学习', '玩', '吃饭', '睡觉', '书', '猫', '狗', '跑', '走', '笑', '水', '火']
    vocab = {'[PAD]': 0, '[MASK]': 1, '[CLS]': 2, '[SEP]': 3}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def random_sentence(vocab, lo=2, hi=4):
    words = [w for w in vocab if w not in ('[PAD]', '[MASK]', '[CLS]', '[SEP]')]
    n = random.randint(lo, hi)
    return random.sample(words, n)


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


# ------------------------- 模型组件 -------------------------
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


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_size, ffn_hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden)
        self.fc2 = nn.Linear(ffn_hidden, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class AddNorm(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.ln(x + self.dropout(y))


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


class PositionalEmbedding(nn.Module):
    """可学习的位置编码。"""
    def __init__(self, max_len, hidden_size):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, max_len, hidden_size))

    def forward(self, x):
        return x + self.embedding[:, :x.size(1), :]


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


# ------------------------- 训练 -------------------------
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


if __name__ == '__main__':
    train()
