from openai import OpenAI
from chatbot.prompt_template import get_prompt_for_recipe
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize OpenAI client with your API key
client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"), 
    base_url = "https://api.deepseek.com"
)

def call_deepseek_model(risk_score: float, user_data: dict, user_prompt="I want to eat healthy and lose weight. I prefer meals that are low in carbs and high in protein.") -> str:
    health_info = "\n".join([f"{k}: {v}" for k, v in user_data.items()])
    combined_prompt = (
        f"Here is the user's health profile:\n{health_info}\n"
        f"Diabetes risk score: {risk_score:.2f}\n\n"
        f"Now, based on the following preference:\n\"{user_prompt}\"\n"
        f"Please analyze and respond as instructed."
    )
    
    prompt = get_prompt_for_recipe()

    response = client.chat.completions.create(
        model="deepseek-chat", 
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": combined_prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    result = call_deepseek_model("I want MEAT! AS MORE AS POSSIBLE!")
    print(result)