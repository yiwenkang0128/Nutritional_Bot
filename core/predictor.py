
def predict_diabetes_risk(user_data: dict) -> float:
    # 示例：假装计算一下风险分数
    glucose = user_data.get("Glucose", 100)
    bmi = user_data.get("BMI", 25)
    age = user_data.get("Age", 40)
    score = 0.3 * glucose + 0.2 * bmi + 0.1 * age
    return min(score / 100, 1.0)