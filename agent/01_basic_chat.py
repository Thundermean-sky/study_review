"""
Step 1: 最简对话调用
====================
目标：理解 OpenAI 兼容 API 的基本形态，为后面加上工具做准备。

运行前先设置 API Key（Windows PowerShell）：
    $env:DEEPSEEK_API_KEY = "sk-你的key"

DeepSeek 提供的是 OpenAI 兼容接口，所以直接使用 openai 这个包，
只需要把 base_url 换成 DeepSeek 的地址即可。
"""

import os
from openai import OpenAI

# 1. 创建客户端：指向 DeepSeek
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 2. 组装 messages：多轮对话的本质就是维护一个消息列表
messages = [
    {"role": "system", "content": "你是一个天气助手，回答要简洁。"},
    {"role": "user", "content": "北京今天天气怎么样？"},
]

# 3. 发起一次请求
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
)

# 4. 拿到模型回复
answer = response.choices[0].message.content
print("=== 模型回答 ===")
print(answer)
