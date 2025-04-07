# chatbot/chatbot_ui.py
import gradio as gr
from core.predictor import predict_diabetes_risk
from core.recipe_selector import recommend_recipes  # 可选
from chatbot.llm_inference import call_deepseek_model

# 假设用户输入的字段为简化结构，我们使用字典模拟结构化数据

def handle_user_input(age, gender, height, weight, activity, pregnancies, glucose, bp, skin_thickness, insulin):
    height_m = float(height) / 100  
    bmi = float(weight) / (height_m ** 2)
    user_data = {
        "Age": int(age),
        "Gender": gender,
        "Height": float(height),
        "Weight": float(weight),
        "ActivityLevel": activity,
        "Pregnancies": int(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(bp),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": round(bmi, 2)
    }

    risk_score = predict_diabetes_risk(user_data)
    recipes = recommend_recipes(user_data)
    full_response = call_deepseek_model(risk_score, user_data, recipes)

    return full_response

with gr.Blocks() as demo:
    gr.Markdown("## 🩺 糖尿病风险预测助手")

    with gr.Row():
        age = gr.Textbox(label="Age", value="")
        gender = gr.Dropdown(label="Gender", choices=["None","male", "female"])
        height = gr.Textbox(label="Height (cm)", value="")
        weight = gr.Textbox(label="Weight (kg)", value="")

    with gr.Row():
        activity = gr.Dropdown(label="Activity Level", choices=["None","Very Active", "Moderately Active", "Lightly Active", "Sedentary"])
        pregnancies = gr.Textbox(label="Pregnancies", value="")
        glucose = gr.Textbox(label="Glucose", value="")
        bp = gr.Textbox(label="Blood Pressure", value="")

    with gr.Row():
        skin_thickness = gr.Textbox(label="Skin Thickness", value="")
        insulin = gr.Textbox(label="Insulin", value="")

    submit_btn = gr.Button("开始预测")
    output_text = gr.Textbox(label="结果输出", lines=8)

    submit_btn.click(
        fn=handle_user_input,
        inputs=[age, gender, height, weight, activity, pregnancies, glucose, bp, skin_thickness, insulin],
        outputs=output_text
    )


if __name__ == "__main__":
    demo.launch()