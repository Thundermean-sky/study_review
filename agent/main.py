import json
import os
from openai import OpenAI

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_meat_price",
            "description": "查询某个城市的肉类价格，当用户询问肉价时必须调用此工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：四川"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。当用户询问‘现在几点？’、‘今天几号’等时间问题时必须调用",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def get_meat_price(city: str) -> str:
    table = {
        "重庆": "今日猪肉价格：12 元/斤， 羊肉价格：25 元/斤， 牛肉价格：55 元/斤",
        "四川": "今日猪肉价格：11 元/斤， 羊肉价格：15 元/斤， 牛肉价格：35 元/斤",
        "北京": "今日猪肉价格：22 元/斤， 羊肉价格：35 元/斤， 牛肉价格：65 元/斤"
    }

    return table.get(city, f"暂无{city}的肉价，请稍微再询问")

def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


TOOL_MAP = {
    "get_meat_price": get_meat_price,
    "get_current_time": get_current_time
}

client = OpenAI(
    api_key='sk-bf825dd6818842b19dd000785095a62d',
    base_url='https://api.deepseek.com'
)

messages = [
    {
        "role": "system",
        "content": "你是一个优秀的助手。"
    },
    {
        "role": "user",
        "content": "重庆和四川今天各类肉的价格如何？另外现在几点了呀？"
    }
]

# MAX_TURNS = 5


def run_agent(client, messages, tools, tool_map, model='deepseek-v4-flash', max_turns=5):
    for t in range(max_turns):
        print(f"Current turn: {t+1}/{max_turns}")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort='low',
            extra_body={
                "type": {
                    "thinking": "disabled"
                }
            },
            tools=tools
        )

        message = response.choices[0].message

        print("======message====")
        print(message)

        if not message.tool_calls:
            print("==Agent最终回答==")
            print(message.content)
            return message.content

        messages.append(message)




        for call in message.tool_calls:
            func_name = call.function.name
            try:
                func_args = json.loads(call.function.arguments)
                res = tool_map[func_name](**func_args)
            except KeyError:
                res = f"Error {func_name}"
            except Exception as e:
                res = f"Error {func_name} 执行失败”{e}"


            print(f"Agent 调用{func_name} ({func_args})")
            print(f"[Tool]调用结果  {res}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": res
            })

    print(f"over max turns: {max_turns}")
    return None


answer = run_agent(client, messages, tools, TOOL_MAP, model='deepseek-v4-flash', max_turns=5)

print(messages)
