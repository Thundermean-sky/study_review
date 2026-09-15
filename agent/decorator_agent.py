import json
import os
from openai import OpenAI

TOOL_MAP = {}
TOOL_SCHEMAS = []


def register_tool(name, description, parameters=None):
    """装饰器：注册工具函数"""

    def decorator(f):
        TOOL_MAP[name] = f
        TOOL_SCHEMAS.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters if parameters is not None else {"type": "object", "properties": {}}
            }
        })
        return f

    return decorator


@register_tool(
    name="get_meat_price",
    description="查询某个城市的肉类价格，当用户询问肉价时必须调用",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称，例如：四川"}
        },
        "required": ["city"],
    }
)
def get_meat_price(city: str) -> str:
    table = {
        "重庆": "今日猪肉价格：12 元/斤， 羊肉价格：25 元/斤， 牛肉价格：55 元/斤",
        "四川": "今日猪肉价格：11 元/斤， 羊肉价格：15 元/斤， 牛肉价格：35 元/斤",
        "北京": "今日猪肉价格：22 元/斤， 羊肉价格：35 元/斤， 牛肉价格：65 元/斤",
    }
    return table.get(city, f"暂无{city}的肉价，请稍微再询问")


@register_tool(
    name="get_current_time",
    description="获取当前的日期和时间。当用户询问现在几点、今天几号等时间问题时必须调",
    parameters={
        "type": "object",
        "properties": {}
    }
)
def get_current_time() -> str:
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time


@register_tool(
    name="record_info",
    description="适当的总结并记录用户每一次的对话内容",
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "用户的的对话记录"},
        },
        "required": ["content"],
    }
)
def record_info(content: str) -> str:
    with open("note.txt", "a", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")
        f.close()
    return f"已记录"


@register_tool(
    name='get_file_size',
    description="获取文件的大小",
    parameters={
        "type": "object",
        "properties": {
            "file_name": {"type": "string", "description": "需要计算大小的文件名称"},
        },
        "required": ["file_name"],
    }
)
def get_file_size(file_name: str) -> str:
    with open(file_name, "r", encoding="utf-8") as f:
        data = f.read()

    return str(len(data))

client = OpenAI(
    api_key='sk-bf825dd6818842b19dd000785095a62d',
    base_url='https://api.deepseek.com'
)


def run_agent(messages, model='deepseek-v4-flash', max_turn=5):
    """循环引擎：内部自动读取 TOOL_SCHEMAS / TOOL_MAP, 业务层无需关心"""
    for t in range(max_turn):
        print(f"Current turn: {t}")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort='low',
            extra_body={
                'type': {
                    "thinking": "disabled"
                }
            },
            tools=TOOL_SCHEMAS
        )

        message = response.choices[0].message

        messages.append(message)
        print(message)

        if not message.tool_calls:
            print("Final Answer")
            print(message.content)
            return message.content

        for call in message.tool_calls:
            func_name = call.function.name
            try:
                func_args = json.loads(call.function.arguments)
                res = TOOL_MAP[func_name](**func_args)

            except KeyError:
                print(f"TOOL {func_name} 不存在， 可用列表{list(TOOL_MAP)}")

            except Exception as e:
                print(f"TOOL {func_name} 调用失败")

            print(f"Agent 调用 {func_name} ({func_args})")
            print(f"result : {res}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": res
            })
    print(f"over max turn: {max_turn}")
    return None


messages = [
    {"role": "system", "content": "你是一个优秀的助手。当用户询问文件大小时，必须调用对应工具获取数据后再回答。同时记录用户每一轮的你和用户的对话内容"},
    {"role": "user", "content": "帮我算一下note.txt这个文件的大小"},
]

# print("Auto collect tools schema: ", json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2))

run_agent(messages)
