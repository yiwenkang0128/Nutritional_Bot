# chatbot/chatbot_ui.py
import gradio as gr
import numpy as np
import pandas as pd
from core.induce import predict
from chatbot.llm_inference import call_deepseek_model
from MCP.utils.ParseJSON import parse_json
from MCP.utils.recipe_recommender import recommend_recipes_for_user

RECIPE_DF_FULL = pd.read_csv("MCP/data/diabetic_recipes_with_categories.csv")

def handle_user_input(age, gender, height, weight, activity, pregnancies, glucose,diet_preference):
    try:
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
            "BMI": round(bmi, 2)
        }
        
        diabetes_flag = np.array([
                user_data["Pregnancies"],
                user_data["Glucose"],
                user_data["BMI"],
                user_data["Age"]
        ])

        #Call the ML model to get the diabetes risk score.
        prediction_result = predict(diabetes_flag)
        risk_score = prediction_result['soft_vote']

        #Call the LLM to get dietary recommendations in JSON format.
        llm_result = call_deepseek_model(risk_score, user_data, diet_preference)
        

        #Parse the LLM result into a dictionary, get the category scores.
        category_scores = parse_json(llm_result)

        #Call the recipe recommender to get the recipes.
        recipes = recommend_recipes_for_user(category_scores, risk_score)

        #Summary text
        if risk_score < 0.5:
            summary = f"✅ Congratulations, your risk of diabetes is low, with a score of only **{risk_score:.2f}**。\nHere are the recommended recipes for you:"
        else:
            summary = f"⚠️ Attention, your risk of diabetes is high **{risk_score:.2f}**。\nHere are some recipes you may refer to with caution:"

        #Build Accordion
        recipe_list = []
        for _, row in recipes.iterrows():
            name = row["recipeName"]
            tag = "（Raises blood sugar relatively more—consume in small amounts.）" if (risk_score >= 0.5 and row["diabetic_note"] == "not suitable") else ""
            title = f"{name} {tag}"

            matched = RECIPE_DF_FULL[RECIPE_DF_FULL["recipeName"] == name]
            if not matched.empty:
                steps_raw = matched.iloc[0]["steps"]
                steps_list = eval(steps_raw) if isinstance(steps_raw, str) else ["No steps information"]
                steps_md = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps_list)])
            else:
                steps_md = "❗ Sorry, we could not find the recipe steps."

            recipe_list.append((title, steps_md))

        recipe_blocks = [f"### {title}\n\n{steps_md}" for title, steps_md in recipe_list]
        recipe_blocks += [""] * (5 - len(recipe_blocks))

        return summary, *recipe_blocks

    except Exception as e:
        error_text = f"Error: {str(e)}"
    return error_text, *[""] * 5 

with gr.Blocks() as demo:
    gr.Markdown("# 🩺 糖尿病风险预测助手")

    with gr.Row():
        age = gr.Textbox(label="Age", value="")
        gender = gr.Dropdown(label="Gender", choices=["None","male", "female"])
        height = gr.Textbox(label="Height (cm)", value="")
        weight = gr.Textbox(label="Weight (kg)", value="")

    with gr.Row():
        activity = gr.Dropdown(label="Activity Level", choices=["None","Very Active", "Moderately Active", "Lightly Active", "Sedentary"])
        pregnancies = gr.Textbox(label="Pregnancies", value="")
        glucose = gr.Textbox(label="Glucose(mg/dL)", value="")


    with gr.Row():
        diet_input = gr.Textbox(label="输入你的饮食想法", placeholder="例如：I want to eat healthy and low carb. I love beef.")


    submit_btn = gr.Button("Start")

    result_summary = gr.Markdown()
    recipe_outputs = [gr.Markdown(visible=True) for _ in range(5)]  

    submit_btn.click(
        fn=handle_user_input,
        inputs=[age, gender, height, weight, activity, pregnancies, glucose, diet_input],
        outputs=[result_summary] + recipe_outputs
    )


def main():
    demo.launch()

