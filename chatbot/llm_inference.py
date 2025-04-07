from openai import OpenAI
from prompt_template import get_prompt_for_recipe
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize OpenAI client with your API key
client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"), 
    base_url = "https://api.deepseek.com"
)

def call_deepseek_model(userPrompt="I want to eat healthy and lose weight. I prefer meals that are low in carbs and high in protein.") -> str:
    prompt = get_prompt_for_recipe()

    response = client.chat.completions.create(
        model="deepseek-chat", 
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": userPrompt},
        ],
        stream=False
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    result = call_deepseek_model("I want MEAT! AS MORE AS POSSIBLE!")
    print(result)