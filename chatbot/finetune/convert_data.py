from datasets import load_dataset
import json
import os

# 自动创建输出目录
os.makedirs("chatbot/finetune", exist_ok=True)

dataset = load_dataset("sarthak-wiz01/nutrition_dataset", split="train")

#转换为指令微调格式
def convert_to_messages(record):
    instruction = f"""Based on the following user info, provide a full-day meal recommendation:
- Age: {record['Age']}
- Gender: {record['Gender']}
- Height: {record['Height']} cm
- Weight: {record['Weight']} kg
- Activity Level: {record['Activity Level']}
- Dietary Preference: {record['Dietary Preference']}"""

    response = f"""- Breakfast: {record['Breakfast Suggestion']}
- Lunch: {record['Lunch Suggestion']}
- Dinner: {record['Dinner Suggestion']}
- Snack: {record['Snack Suggestion']}"""

    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
    }

# 生成 JSONL 文件
with open("chatbot/finetune/train_data.jsonl", "w", encoding="utf-8") as f:
    for record in dataset:
        item = convert_to_messages(record)
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Successfully saved: chatbot/finetune/train_data.jsonl")
