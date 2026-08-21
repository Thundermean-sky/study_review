import torch
from torch.utils.data import DataLoader

from tokenizer import CharTokenizer
from data import ToyDataset, collate_fn
from model import LLaVa

def train(epochs=30, batch_size=2, lr=1e-3, seed=0):
    torch.manual_seed(seed)

    tok = CharTokenizer()
    model = LLaVa(tok)

    for p in model.vision_tower.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable, lr)

    ds = ToyDataset(tok, 200, seed)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tok))

    model.train()

    step = 0

    for epoch in range(epochs):
        for images, input_ids, labels in loader:
            _, loss = model(images, input_ids, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if step % 5 == 0:
                print(f"step {step:4d} loss = {loss.item():.4f}")
            step += 1

            if step >= epochs:
                print(f"finished, final loss = {loss.item():.4f}")
                torch.save(model.state_dict(), 'llava_toy.pt')
                return model

    torch.save(model.state_dict(), 'llava_toy.pt')
    return model


if __name__ == '__main__':
    train()