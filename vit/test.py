import torch
from torchvision import transforms
from PIL import Image

from modules.transformers import vit_base_patch16_224

# 1. 构造模型并切换到评估模式（不训练，关掉 dropout 等）
model = vit_base_patch16_224(num_classes=1000)
model.eval()


# ---------- 方式一：随机张量，快速验证架构能否跑通 ----------
# 模型要求输入 (B, C, H, W)，且 H=W=img_size(224)
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():          # 不计算梯度，省显存/内存
    out = model(dummy)
print("随机张量 -> 输出 shape:", tuple(out.shape))   # 期望 (1, 1000)


# ---------- 方式二：真实图片，走完整预处理流程 ----------
# IMG_PATH = "test.jpg"          # 改成你的图片路径，放在本目录下或写绝对路径

# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225]),
# ])

# img = Image.open(IMG_PATH).convert("RGB")   # 转成 3 通道
# x = preprocess(img).unsqueeze(0)            # 加 batch 维 -> (1, 3, 224, 224)

# with torch.no_grad():
#     out = model(x)

# print("真实图片 -> 输出 shape:", tuple(out.shape))
# probs = torch.softmax(out, dim=1)
# top5 = torch.topk(probs, 5)
# print("Top-5 概率:", top5.values.squeeze().tolist())
# print("Top-5 类别索引:", top5.indices.squeeze().tolist())
