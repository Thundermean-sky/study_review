SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<image>", "<unk>"]
CHARS = [chr(c) for c in range(32, 127)]


class CharTokenizer:
    def __init__(self):
        self.special_tokens = SPECIAL_TOKENS
        self.chars = CHARS

        self.special2id = {t: i for i, t in enumerate(self.special_tokens)}

        self.char2id = {c: i + len(self.special_tokens) for i, c in enumerate(self.chars)}

        self.id2char = {v: k for k, v in self.char2id.items()}

        self.pad_id = self.special2id['<pad>']
        self.bos_id = self.special2id['<bos>']
        self.eos_id = self.special2id['<eos>']
        self.image_id = self.special2id['<image>']
        self.unk_id = self.special2id['<unk>']

        self.vocab_size = len(self.special_tokens) + len(self.chars)

    def encode(self, text: str) -> list[int]:
        ids = []
        i, n = 0, len(text)
        while i < n:
            matched = False
            for tok in self.special_tokens:
                if text.startswith(tok, i):
                    ids.append(self.special2id[tok])
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            ids.append(self.char2id.get(ch, self.unk_id))
            i += 1

        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for idx in ids:
            if idx < len(self.special_tokens) and idx != -100:
                out.append(self.special_tokens[idx])
            elif idx in self.id2char:
                out.append(self.id2char[idx])
            else:
                out.append("<unk>")

        return "".join(out)



if __name__ == '__main__':
    tok = CharTokenizer()
    print("voc size:  ", tok.vocab_size)

    ids = tok.encode("<bos><image><image>USER: What shape? ASSISTANT: red circle<eos>")

    print("ids =  ", ids)

    print("decode(ids) = ", tok.decode(ids))