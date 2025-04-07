from openai import OpenAI
from .prompt_template import get_prompt
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"), 
    base_url = "https://api.deepseek.com"
)

def call_deepseek_model(risk_score: float, user_data: dict, recipes: list[str]) -> str:
    prompt = get_prompt(risk_score, user_data, recipes)

    response = client.chat.completions.create(
        model="deepseek-chat",  # 可换成 deepseek-coder 等
        messages=[
            {"role": "system", "content": "你是一位营养专家，擅长为用户解读健康数据。"},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )

    return response.choices[0].message.content
