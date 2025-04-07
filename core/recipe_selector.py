def recommend_recipes(user_data: dict) -> list[str]:
    # 假设根据 BMI、Glucose 等字段做简单筛选
    recipes = []

    if user_data["Glucose"] > 140:
        recipes.append("蒸豆腐配蔬菜（低GI、高蛋白）")
    if user_data["BMI"] > 28:
        recipes.append("番茄鸡胸肉沙拉（高纤低脂）")
    if user_data["Age"] > 50:
        recipes.append("黑木耳炒鸡蛋（抗氧化）")

    # 固定几道常规菜
    recipes += ["糙米饭", "清炒苦瓜", "小米粥"]

    return recipes
