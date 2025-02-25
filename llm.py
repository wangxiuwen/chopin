from openai import OpenAI
import os
from config import ConfigLoader  # 导入配置加载器

cfg = ConfigLoader().config

class LLM:
    def __init__(self, api_key=None, base_url=None, model=None):
        # 优先使用传入参数，其次使用配置文件参数
        print(cfg)
        self.client = OpenAI(
            api_key=api_key or cfg.llm.api_key,
            base_url=base_url or cfg.llm.base_url,
        )
        self.model = model or cfg.llm.model

    
    def call(self, messages):
         
        completion = self.client.chat.completions.create(
            model="qwen-plus",
            messages=messages
        )
        return completion.choices[0].message.content


if __name__ == '__main__':
    # 使用示例（完全从配置文件读取）
    llm = LLM()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}
    ]
    response = llm.call(messages)
    print(response)