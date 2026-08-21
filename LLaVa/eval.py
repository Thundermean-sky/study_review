import torch

from tokenizer import CharTokenizer
from model import LLaVa
import random
from data import make_img, SHAPES, COLORS, POSITIONS


def generate(model, tok, image, max_new=40):
    prompt = "<bos><image>USER: What is the shape and color? ASSISTANT: "
    ids = tok.encode(prompt)

    with torch.no_grad():
        for _ in range(max_new):
            input_ids = torch.tensor([ids])
            logits = model(image.unsqueeze(0), input_ids)
            next_id = logits[0, -1].argmax().item()
            ids.append(next_id)
            if next_id == tok.eos_id:
                break


    return tok.decode(ids)


def evaluate(num_test=10, ckpt=""):
    tok = CharTokenizer()
    model = LLaVa(tok)

    model.load_state_dict(torch.load(ckpt, map_location='cuda'))

    random.seed(1)

    correct = 0
    for i in range(num_test):
        shape = random.choice(list(SHAPES))
        color = random.choice(list(COLORS))
        position = random.choice(POSITIONS)
        image = torch.tensor(make_img(shape, color, position))

        answer = f"{shape} {color}"

        out = generate(model, tok, image)

        pred = out.split("ASSISTANT: ")[-1].replace("<eos>", "").strip()

        ok = pred == answer

        correct += ok
        print(f"test: {i}: 真实={answer:14s} 预测={pred:14s}")

    print(f"accuracy: {correct / num_test:14s}")

if __name__ == '__main__':
    evaluate()