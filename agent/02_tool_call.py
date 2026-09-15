"""
Step 2+3: 完整的工具调用循环（核心！）
======================================
把 Step 1 讲过的原理写成代码：

  模型输出工具调用指令 -> 程序解析并执行 -> 结果回传 -> 模型再推理 -> ...循环
  直到模型不再输出 tool_calls，直接给出最终回答。

运行前先设置 API Key（Windows PowerShell）：
    $env:DEEPSEEK_API_KEY = "sk-你的key"
"""

import json
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ==================================================================
# STEP 2: 定义工具（Tools Schema）
# ------------------------------------------------------------------
# 这里用的是 JSON Schema，只描述工具"长什么样、参数是什么"，
# 模型读到这份描述后，就知道该在什么时候、用什么参数来调用它。
# 注意：这里只是"说明书"，不是真正的实现。
# ==================================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的当前天气情况。当用户询问天气时必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

# ==================================================================
# 工具的真实实现（本地 Python 函数）
# ------------------------------------------------------------------
# 这才是"手"真正干活的地方。真实项目里，这里可以调第三方天气 API、
# 查数据库、读写文件等。我们先用一个字典模拟。
# ==================================================================
def get_weather(city: str) -> str:
    """模拟天气查询。"""
    table = {
        "北京": "晴，25°C，东南风 2 级",
        "上海": "多云，28°C，湿度 70%",
        "广州": "雷阵雨，30°C，出行请带伞",
    }
    return table.get(city, f"暂无 {city} 的天气数据，返回默认值：晴，22°C")

# 工具名 -> 函数 的映射表。这是 Agent 的"手"，用来按名字找到要执行的函数。
TOOL_MAP = {
    "get_weather": get_weather,
}

# ==================================================================
# 组装对话
# ==================================================================
messages = [
    {"role": "system", "content": "你是一个天气助手。当用户询问天气时，必须调用 get_weather 工具获取数据后再回答。"},
    {"role": "user", "content": "北京和上海今天天气怎么样？"},
]

# ==================================================================
# STEP 3: Agent 循环
# ==================================================================
MAX_TURNS = 5  # 防止意外死循环，最多允许 5 轮工具调用

for turn in range(MAX_TURNS):
    print(f"\n---------- 第 {turn + 1} 次请求模型 ----------")

    # 每次请求都把完整的历史消息发过去
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,          # 告诉模型：你可以用这些工具
    )
    message = response.choices[0].message

    # 情况 A：模型没有要求调用工具 -> 直接给出了最终回答，循环结束
    if not message.tool_calls:
        print("=== Agent 最终回答 ===")
        print(message.content)
        break

    # 情况 B：模型要求调用工具 -> 它输出的是"指令"，不是结果！
    # 先把这条带 tool_calls 的消息追加进历史（这是 OpenAI 协议的要求）
    messages.append(message)

    for call in message.tool_calls:
        func_name = call.function.name
        func_args = json.loads(call.function.arguments)  # arguments 是 JSON 字符串，需要解析

        print(f"[Agent 指令] 调用工具: {func_name}({func_args})")

        # 真正的执行发生在这里（模型不执行，代码执行）
        result = TOOL_MAP[func_name](**func_args)

        print(f"[Tool 结果] {result}")

        # 把工具结果以 role="tool" 的消息回传给模型
        # tool_call_id 必须和上面的调用 id 对应，模型靠它把结果和指令配对
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result,
        })

    # 继续下一轮：模型看到工具结果后，会综合推理给出最终答案
