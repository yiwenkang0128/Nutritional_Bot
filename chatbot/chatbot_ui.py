# chatbot/chatbot_ui.py
import gradio as gr
import numpy as np
import pandas as pd
from core.induce import predict
from chatbot.llm_inference import call_deepseek_model
from MCP.utils.ParseJSON import parse_json
from MCP.utils.recipe_recommender import recommend_recipes_for_user

RECIPE_DF_FULL = pd.read_csv("MCP/data/diabetic_recipes_with_categories.csv")

def format_recipe_for_display(recipe_row):
    """
    Correctly formats a recipe's details into proper Markdown for the UI display
    
    Args:
        recipe_row: A pandas Series or dict containing recipe data
        
    Returns:
        dict: A dictionary with formatted title, ingredients, and steps
    """
    name = recipe_row["recipeName"]
    
    # Get serving information
    serves = recipe_row.get("serves", "Unknown")
    serves_text = f"**Servings:** {serves} people" if serves else "**Servings:** Not provided"
    
    # Format ingredients as a proper Markdown list
    ingredients_md = []
    try:
        ingredients_raw = recipe_row.get("ingredients", "[]")
        # Clean original string, remove extra quotes and spaces
        ingredients_raw = ingredients_raw.replace("'", "").replace("[", "").replace("]", "")
        # Split ingredients by actual delimiter
        ingredients_list = [ing.strip() for ing in ingredients_raw.split('\n') if ing.strip()]
        
        if ingredients_list:
            for ing in ingredients_list:
                # Process each ingredient item
                clean_ing = ing.strip()
                if '(' in clean_ing and ')' in clean_ing:
                    # Keep bracket content and main content on the same line
                    ingredients_md.append(f"- {clean_ing}")
                else:
                    # Split other cases by space
                    parts = clean_ing.split()
                    if len(parts) > 1:
                        # Keep quantity units and ingredient names together
                        ingredients_md.append(f"- {' '.join(parts)}")
                    else:
                        ingredients_md.append(f"- {clean_ing}")
        else:
            ingredients_md = ["- No ingredients information"]
    except Exception as e:
        ingredients_md = [f"- Failed to parse ingredients: {str(e)}"]
    
    # Format steps as a numbered Markdown list
    steps_md = []
    try:
        steps_raw = recipe_row.get("steps", "[]")
        # Clean original string, remove quotes and brackets
        steps_raw = steps_raw.replace("'", "").replace("[", "").replace("]", "")
        # Split into different steps by newline
        steps_list = [step.strip() for step in steps_raw.split('\n') if step.strip()]
        
        if steps_list:
            step_number = 1
            for step in steps_list:
                # Clean step text
                clean_step = step.strip()
                # Split substeps by period
                sub_steps = [s.strip() for s in clean_step.split('.') if s.strip()]
                
                for sub_step in sub_steps:
                    # Ensure first letter is capitalized
                    sub_step = sub_step.capitalize()
                    # Add step number and newline
                    steps_md.append(f"{step_number}. {sub_step}")
                    step_number += 1
        else:
            steps_md = ["1. No cooking steps available"]
    except Exception as e:
        steps_md = [f"❗ Failed to parse cooking steps: {str(e)}"]
    
    # Combine everything into the proper format
    ingredients_section = "\n".join(ingredients_md)
    steps_section = "\n\n".join(steps_md)
    
    formatted_content = f"""{serves_text}

**Ingredients:**

{ingredients_section}

**Cooking Steps:**

{steps_section}
"""
    
    return {
        "title": f"### 🍽 {name}",
        "content": formatted_content
    }

def handle_user_input(age, gender, height, weight, activity, pregnancies, glucose, diet_preference):
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

        # Call the ML model to get the diabetes risk score
        prediction_result = predict(diabetes_flag)
        risk_score = prediction_result['soft_vote']

        # Call the LLM to get dietary recommendations in JSON format
        llm_result = call_deepseek_model(risk_score, user_data, diet_preference)
        # Parse the LLM result into a dictionary, get the category scores
        category_scores = parse_json(llm_result)
        # Call the recipe recommender to get the recipes
        recipes = recommend_recipes_for_user(category_scores, risk_score)

        # Summary text
        if risk_score < 0.5:
            summary = f"✅ Congratulations! Your diabetes risk is low, with a risk score of **{risk_score:.2f}**.\nHere are some recommended recipes:"
        else:
            summary = f"⚠️ Attention! Your diabetes risk is high, with a risk score of **{risk_score:.2f}**.\nHere are some recipes you should consume with caution:"

        # Build recipe displays
        recipe_titles = []
        recipe_bodies = []

        for _, row in recipes.iterrows():
            name = row["recipeName"]
            tag = "（⚠️ May increase blood sugar, please consume in moderation）" if (risk_score >= 0.5 and row.get("diabetic_note") == "not suitable") else ""
            
            matched = RECIPE_DF_FULL[RECIPE_DF_FULL["recipeName"] == name]
            if not matched.empty:
                recipe_row = matched.iloc[0]
                formatted = format_recipe_for_display(recipe_row)
                title = formatted["title"] + " " + tag
                content_md = formatted["content"]
            else:
                title = f"### 🍽 {name} {tag}"
                content_md = "❗ Sorry, we couldn't find detailed information for this recipe."
                
            recipe_titles.append(title)
            recipe_bodies.append(content_md)

        # 填充空数据直到 5 个
        while len(recipe_titles) < 5:
            recipe_titles.append("")
            recipe_bodies.append("")

        # 将标题和正文分别打包成 (value, visible) 的形式
        result_blocks = []
        for i in range(5):
            if recipe_titles[i].strip():
                title = recipe_titles[i]
                content = recipe_bodies[i]
                show = True
            else:
                title = ""
                content = ""
                show = False

            result_blocks.extend([
                gr.update(value=title, visible=show),  # 标题
                gr.update(value=content, visible=show),  # 内容
                gr.update(visible=show)  # 组
            ])
            
        return [summary] + result_blocks
        
    except Exception as e:
        # 错误情况下也要返回正确数量的空值
        empty_updates = [gr.update(value="", visible=False) for _ in range(10)] + [gr.update(visible=False) for _ in range(5)]
        return [f"Error: {str(e)}"] + empty_updates

# UI Component setup
with gr.Blocks(css="""
    body, .gradio-container { background-color: #121212; color: #e0e0e0; }
    .container { max-width: 950px; margin: auto; }
    .header { text-align: center; margin-bottom: 30px; color: #81c1fa; }
    .input-section { 
        margin-bottom: 25px; 
        background-color: #1e1e1e; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #333;
    }
    .recipe-box { 
        margin-bottom: 20px; 
        background-color: #1e1e1e; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.2); 
        border-left: 5px solid #3498db;
    }
    .summary-highlight { 
        padding: 15px; 
        background-color: #252525; 
        border-radius: 12px; 
        margin: 20px 0; 
        border-left: 5px solid #3498db;
        font-size: 16px;
    }
    button.primary { 
        background-color: #2980b9; 
        color: white;
    }
    button.primary:hover { 
        background-color: #3498db; 
    }
    label, .input-label { color: #aaa !important; }
    input, select, textarea { 
        background-color: #333 !important; 
        border: 1px solid #555 !important; 
        color: #e0e0e0 !important;
    }
    .markdown-body h3 { color: #81c1fa; }
    .markdown-body strong { color: #81c1fa; }
    .recipe-box ul { margin-top: 0.5em; }
    .recipe-box li { margin-bottom: 0.3em; }
""") as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown("# 🩺 Diabetes Risk Prediction and Diet Assistant", elem_classes=["header"])
        
        with gr.Column(elem_classes=["input-section"]):
            gr.Markdown("### Please Enter Your Personal Information")
            with gr.Row():
                age = gr.Textbox(label="Age", value="")
                gender = gr.Dropdown(label="Gender", choices=["None","male", "female"])
                height = gr.Textbox(label="Height (cm)", value="")
                weight = gr.Textbox(label="Weight (kg)", value="")

            with gr.Row():
                activity = gr.Dropdown(label="Activity Level", choices=["None","Very Active", "Moderately Active", "Lightly Active", "Sedentary"])
                pregnancies = gr.Textbox(label="Number of Pregnancies", value="")
                glucose = gr.Textbox(label="Blood Glucose (mg/dL)", value="")

            with gr.Row():
                diet_input = gr.Textbox(label="Dietary Preferences", placeholder="Example: I like low-carb food; I like beef; I want to eat healthy and lose weight...")

            submit_btn = gr.Button("Start Analysis", variant="primary", size="lg")

        result_summary = gr.Markdown(elem_classes=["summary-highlight"])

        recipe_titles_md = []
        recipe_contents_md = []
        recipe_blocks = []

        for _ in range(5):
            with gr.Group(visible=False, elem_classes=["recipe-box"]) as recipe_block:
                title_md = gr.Markdown()
                content_md = gr.Markdown()
            recipe_titles_md.append(title_md)
            recipe_contents_md.append(content_md)
            recipe_blocks.append(recipe_block)

        outputs = [result_summary]
        for i in range(5):
            outputs.append(recipe_titles_md[i])
            outputs.append(recipe_contents_md[i])
            outputs.append(recipe_blocks[i])

        submit_btn.click(
            fn=handle_user_input,
            inputs=[age, gender, height, weight, activity, pregnancies, glucose, diet_input],
            outputs=outputs
        )

def run_demo():
    demo.launch()

if __name__ == "__main__":
    run_demo()