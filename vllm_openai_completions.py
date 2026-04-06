from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx",
)

chat_outputs = client.chat.completions.create(
    model="Qwen3-8B",
    messages=[{
        "role": "user",
        "content": "什么是深度学习？<think>\n"    # 结尾加上<think>\n启用思考模式
    }]
)

print(chat_outputs.choices[0].message)