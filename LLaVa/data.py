import numpy as np
import torch
from tokenizer import CharTokenizer


SHAPES = {"circle", "square", "triangle",}
COLORS = {"red": (220, 50, 50), "green": (50, 180, 70), "blue": (50, 90, 220),}
POSITIONS = ['left', 'right', 'center']

def make_img(shape, color, position, size=336, margin=40, r=40):
    img = np.full((3, size, size), 225, dtype=np.float32)

    yc, xc = size // 2, size // 2
    if position == "left":
        xc -= size // 4
    elif position == "right":
        xc += size // 4

    c = COLORS[color]
    yy, xx = np.mgrid[0:size, 0:size]

    if shape == "circle":
        mask = (xx - xc) ** 2 + (yy - yc) ** 2 <= r ** 2
    elif shape == "square":
        mask = (abs(xx - xc) <= r) & (abs(yy - yc) <= r)
    else:  # triangle
        mask = (yy - yc) <= (r - abs(xx - xc))              # 占位，稍后修正
    for ch in range(3):
        img[ch][mask] = c[ch]
    return img / 255.0  # 归一化到 [0,1]


def build_sample(tok, shape, color, position):
    question = f"<image>USER: What is the shape and color? ASSISTANT: "
    answer = f"{color} {shape}"
    text = "<bos>" + question + answer + "<eos>"
    input_ids = tok.encode(text)

    labels = [-100] * len(input_ids)
    ans_ids = tok.encode(answer)
    start = len(tok.encode("<bos>" + question))
    labels[start: start + len(ans_ids)] = ans_ids

    return input_ids, labels

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, num_samples=200, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.tok = tokenizer
        self.samples = []
        for _ in range(num_samples):
            shape = rng.choice(list(SHAPES))
            color = rng.choice(list(COLORS))
            position = rng.choice(POSITIONS)
            self.samples.append((shape, color, position))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shape, color, position = self.samples[idx]
        image = make_img(shape, color, position)
        input_ids, labels = build_sample(self.tok, shape, color, position)
        return {
            "image": torch.tensor(image),
            "input_ids": torch.tensor(input_ids,dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }



def collate_fn(batch, tok):
    images = torch.stack([b['image'] for b in batch])
    max_len = max(b['input_ids'].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), tok.pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        input_ids[i, :b['input_ids'].size(0)] = b['input_ids']
        labels[i, :b['labels'].size(0)] = b['labels']

    return images, input_ids, labels

if __name__ == '__main__':
    tokn = CharTokenizer()
    ds = ToyDataset(tokn, num_samples=5)
    images, input_ids, labels = collate_fn([ds[i] for i in range(3)], tokn)

    print("images: ", images.shape)
    print("input_ids: ", input_ids.shape)
    print("labels: ", labels[0].tolist())
    print("Decode: ", tokn.decode(input_ids[0].tolist()))