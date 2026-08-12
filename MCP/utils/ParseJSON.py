import json

def parse_json(content):
    content_str = content
    parsed = json.loads(content_str)
    category_scores = parsed["category_scores"]
    return category_scores

if __name__ == "__main__":
    data = '''
{
  "user_input": "I want MEAT! AS MORE AS POSSIBLE!",
  "category_scores": {
    "meat": 5,
    "high_protein": 5
  },
  "reasoning": "The user's statement strongly emphasizes a desire for meat, indicating a very high preference for the 'meat' category. Since meat is also a primary source of high protein, the 'high_protein' category is also scored highly. Other categories are not mentioned, so they are not included in the response."
}
'''
    parsed = parse_json(data)
    print(parsed)
    print(type(parsed))  # Should be a dictionary
