# chatbot/prompt_template.py
def get_prompt(risk_score: float, user_data: dict, recipes: list[str]) -> str:
    return f"""
你是一位专业营养师，请根据以下健康数据和风险评分，给出风险解释及饮食建议：

健康数据：{user_data}
风险评分：{risk_score:.2f}
推荐食谱：{recipes}

请用简洁自然的语言向用户解释他们的风险，并推荐饮食调整建议。
"""
