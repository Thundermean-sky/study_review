import torch
import torch.nn as nn
import torch.nn.functional as F

from vision import CLIPViT
from projector import MLPProjector
from llm import LLaMA
from tokenizer import CharTokenizer


class LLaVa(nn.Module):
    def __init__(self, tokenizer, img_size=336, patch=14, vis_dim=1024, vis_hidden=4096, vis_depth=4,
                 llm_dim=4096, llm_heads=32, llm_hidden=11008, llm_depth=4, max_seq_len=1024):
        super().__init__()
        self.img_id = tokenizer.image_id
        self.pad_id = tokenizer.pad_id
        self.vocab_size = tokenizer.vocab_size
        self.vision_tower = CLIPViT(img_size=img_size, patch=patch, dim=vis_dim, hidden=vis_hidden, depth=vis_depth)
        self.projector = MLPProjector(vis_dim, llm_dim, llm_dim)

        self.llm = LLaMA(llm_dim, llm_heads, llm_hidden, llm_depth, self.vocab_size, max_seq_len)

    def forward(self, img, input_ids, labels=None):
        B, T = input_ids.shape
        visual = self.projector(self.vision_tower(img))  # (B,576,4096)
        text = self.llm.token_emb(input_ids)  # (B,T,4096)

        # ① 展开文本序列（这一步始终需要，生成时也要）
        embed_list = []
        for b in range(B):
            emb_chunk = []
            for t in range(T):
                if input_ids[b, t].item() == self.img_id:
                    emb_chunk.append(visual[b])  # (576,4096)
                else:
                    emb_chunk.append(text[b, t:t + 1])  # (1,4096)
            embed_list.append(torch.cat(emb_chunk, dim=0))

        max_len = max(e.size(0) for e in embed_list)
        embeds = embed_list[0].new_zeros(B, max_len, self.llm.token_emb.embedding_dim)
        for b in range(B):
            embeds[b, :embed_list[b].size(0)] = embed_list[b]

        logits = self.llm(input_embeds=embeds)  # (B,max_len,100)

        if labels is None:  # 生成模式：不需要 label
            return logits

        # ② 仅训练时：展开 labels（视觉部分填 -100）
        label_list = []
        for b in range(B):
            lab_chunk = []
            for t in range(T):
                if input_ids[b, t].item() == self.img_id:
                    lab_chunk.append(torch.full((576,), -100, dtype=torch.long,
                                                device=input_ids.device))
                else:
                    lab_chunk.append(labels[b, t:t + 1])
            label_list.append(torch.cat(lab_chunk, dim=0))

        padded_labels = labels.new_full((B, max_len), -100)
        for b in range(B):
            padded_labels[b, :label_list[b].size(0)] = label_list[b]

        shift_logits = logits[:, :-1, :].reshape(-1, self.vocab_size)
        shift_labels = padded_labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return logits, loss


if __name__ == '__main__':
    tok = CharTokenizer()
    model = LLaVa(tok)
    print("total params = ", sum(p.numel() for p in model.parameters()))

    image = torch.randn(1, 3, 336, 336)

    ans = "ok"
    text = "<bos><image>USER: hi ASSISTANT: " + ans + "<eos>"
    ids = tok.encode(text) + [tok.pad_id] * 10
    input_ids = torch.tensor([ids])
    labels = torch.full_like(input_ids, -100)
    eos_pos = ids.index(tok.eos_id)
    labels[0, eos_pos - len(tok.encode(ans)): eos_pos] = torch.tensor(tok.encode(ans))

    logits, loss = model(image, input_ids, labels)
    print("logits shape = ", logits.shape)
    print("loss = ", loss.item())
    loss.backward()
    print("done")
