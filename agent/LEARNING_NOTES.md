# DeepSeek Tool Call Agent 学习笔记

> 目标：不依赖 LangChain，用 Python + API 从零理解并实现 Agent 工具调用（Tool Call）。
> 模型：DeepSeek（OpenAI 兼容接口），SDK：`openai`

## 一、核心原理（最重要，先记住）

工具调用最大的认知误区是"模型会执行工具"。真相是：

```
┌─────────────────────────────────────────┐
│ Agent 循环                               │
│                                         │
│  用户提问 ──► 模型（deepseek）             │
│                 │                        │
│        ┌────────┴─────────┐              │
│        ▼                  ▼              │
│   直接回答(文字)      输出工具指令          │
│   ──► 流程结束        {name, arguments}   │
│                          │               │
│                          ▼               │
│                   代码解析并执行真实函数      │
│                          │               │
│                          ▼               │
│             结果以 role="tool" 回传模型    │
│                          │               │
│                    └──► 回到循环顶部       │
└─────────────────────────────────────────┘
```

**一句话：模型负责"想"（下指令），你的代码负责"做"（执行函数），结果再喂回模型"想"。** 模型从不真正执行任何函数。

## 二、分步学习路线

| 步骤 | 文件名 | 学到的内容 |
|------|--------|-----------|
| 1. 最简对话 | `01_basic_chat.py` | OpenAI 兼容 API：`client` + `messages`；多轮对话本质是维护消息列表 |
| 2. 定义工具 | `02_tool_call.py` | **JSON Schema 只是"说明书"**，描述工具名/参数/用途，非实现 |
| 3. 工具循环 | `02_tool_call.py` | `if not tool_calls: 结束`；`TOOL_MAP[名字](**参数)` 真正执行；结果回传 |
| 4. 多工具+防御 | `main.py` | 并行 tool_calls（index 0,1）；错误回传给模型而非崩溃 |
| 5. 封装引擎 | `main.py` | `run_agent()` 参数化，业务与引擎解耦 |
| 6. 装饰器注册 | `decorator_agent.py` | `@register_tool` 合并 schema+实现+注册，LangChain 底层同款思路 |

## 三、协议的 5 条铁律（踩坑总结）

1. **`tool_calls` 是模型下发的指令**：`function.name` + `function.arguments`。
2. **`arguments` 是 JSON 字符串**，必须 `json.loads()` 后才是字典，才能 `**` 展开调用。
3. **消息必须成对配对**：`assistant(tool_calls)` 消息必须 `messages.append(message)` 入史，`tool` 消息必须带对应 `tool_call_id`。少了一行 → 报错 `tool must be a response to a preceding message with 'tool_calls'`。
4. **`tool` 消息的 `content` 必须是字符串**。工具返回 int/bool 等 → 第二轮报错 `content should be a string or a list`。工具函数要保证返回文本，引擎层最好统一 `str(res)` 兜底。
5. **模型只会调用你通过 `tools` 参数递给它的工具**；system 提示词提了但 schema 没给，模型通常不会调用。

## 四、真实 Agent 的工程要点

- **`max_turns` 兜底**：防止模型陷入"反复调用工具"的死循环。
- **工具执行加防御**：模型可能幻觉出不存在的工具名（KeyError），工具内部也可能抛异常。做法是把错误**当成普通结果回传给模型**，让它自我修正。
- **`messages` 是 Agent 的记忆体**：完整历史（含工具调用和结果）每轮全量发给模型 → token 成本线性增长 → 真实产品需要**上下文裁剪/摘要**。
- **装饰器 = 接收函数返回函数的函数**：`@register_tool` 在文件加载时执行，自动把函数登记进 `TOOL_MAP` 和 `TOOL_SCHEMAS`，消除三处重复代码。

## 五、最小可运行骨架（记忆锚点）

```python
messages.append(message)                    # ① 指令入史（不能漏！）
for call in message.tool_calls:             # ② 逐个执行
    args = json.loads(call.function.arguments)
    result = TOOL_MAP[call.function.name](**args)   # ③ 唯一干活的地方
    messages.append({                       # ④ 结果回传（content 必须是 str）
        "role": "tool",
        "tool_call_id": call.id,
        "content": str(result),
    })
```

## 六、进阶路线

1. **接真实世界**：查表 → HTTP API / 数据库 / 读写文件。
2. **看穿框架**：现在读 LangChain 的 `@tool`、`bind_tools()` 源码，会发现就是你写过的模式。
3. **独立自测**：不参考代码，独立写出 `get_file_size` + 让 Agent 调度多个工具完成任务。
4. **流式输出 + 上下文管理**：打字机效果、历史裁剪，是真实产品最常见的两项能力。
