from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx",
)

chat_outputs = client.chat.completions.create(
    model="Qwen3-8B",
    messages=[{
        "role": "user",
        "content": "什么是深度学习？"
    }]
)

print("完整响应对象")
print(chat_outputs)

print("\n" + "=" * 50)
print("模型回复")
print(chat_outputs.choices[0].message.content)

if hasattr(chat_outputs.choices[0].message, 'reasoning_content'):
    reasoning = chat_outputs.choices[0].message.reasoning_content
    if reasoning:
        print("\n思考过程")
        print(reasoning)