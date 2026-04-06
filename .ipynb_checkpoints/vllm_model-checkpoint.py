from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os, json

os.environ['VLLM_USE_MODELSCOPE'] = 'True'

def get_completion(prompts, model, tokenizer=None, temperature=0.6,
                   top_p=0.95, top_k=20, min_p=0, max_tokens=4096,
                   max_model_len=8192):
    stop_token_ids = [151643, 151645]

    # 创建采样参数对象
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )

    # 创建模型
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        max_model_len=max_model_len, # 最大上下文长度，和预分配的KV Cache大小有关
        trust_remote_code=True,      # 是否信任自定义代码，qwen3有自定义的模型代码，需要开启此选项
    )
    # 【创建 LLM 对象时会发生什么？】
    # 1. 读取 config.json，了解模型结构
    # 2. 加载模型权重到 GPU（约 16GB）
    # 3. 编译优化计算图（首次较慢，约 1-2 分钟）
    # 4. 预热 CUDA Graph（捕获不同长度的计算图）
    # 5. 分配 KV Cache（用于存储注意力的中间结果）

    outputs = llm.generate(prompts, sampling_params)

    return outputs


def main():
    model = '/root/autodl-tmp/Qwen/Qwen3-8B'

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    # 读取Qwen3-8B自带的tokenizer.json等文件
    # 其中保存了vocab，merges和各类特殊tokens

    prompt = "给我一个关于大模型的简短介绍。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    print(text)

    outputs = get_completion(text, model)
    print(outputs)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Prompt: {prompt!r}, \nResponse: {generated_text}")

if __name__ == "__main__":
    main()
    