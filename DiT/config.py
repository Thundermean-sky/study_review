import math

# ---- 数据侧 ----
IMAGE_SIZE = 256            # 输入图片边长（像素）
IN_CHANNELS = 3             # RGB
NUM_CLASSES = 10            # 类别数

# ---- 模拟 VAE（latent 空间）----
LATENT_CHANNELS = 4         # 真实 VAE latent 的通道数就是 4
LATENT_SIZE = IMAGE_SIZE // 8   # = 32，模拟 VAE 下采样 8 倍
LATENT_SHAPE = (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)

# ---- Patch 化 ----
PATCH_SIZE = 2              # latent 上每个 patch 的边长
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * LATENT_CHANNELS   # = 16
NUM_PATCHES = (LATENT_SIZE // PATCH_SIZE) ** 2          # = 256，token 总数

# ---- Transformer（简化版）----
HIDDEN_SIZE = 256
DEPTH = 2
NUM_HEADS = 8
MLP_RATIO = 4               # MLP 中间层 = HIDDEN_SIZE * MLP_RATIO
FREQ_EMB_SIZE = 256         # sinusoidal 嵌入的维度

# ---- 扩散 ----
NUM_TIMESTEPS = 1000
