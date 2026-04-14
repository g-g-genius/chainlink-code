import os
from llm import LLMClient, QwenClient

# 方式A：使用OpenAI API
llm = LLMClient(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

# 方式 B：原生接口（需要 enable_search 等专属参数时才用）
llm = QwenClient(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    model="qwen-plus",
    enable_search=True,
)

# Agent 接受任何有 .chat() 方法的对象，无需改动其他代码
agent = Agent(llm=llm)
