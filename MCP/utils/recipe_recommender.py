import pandas as pd

def recommend_recipes_for_user(category_scores, diabetes_probability):
    df = pd.read_csv("MCP/data/diabetic_recipes_with_categories.csv")

    label_cols = list(category_scores.keys())

    # Ensure all label columns are present in the DataFrame
    def compute_weighted_score(row):
        score = 0
        for label in label_cols:
            score += row[label] * category_scores.get(label, 0)
        return score

    df["weighted_score"] = df.apply(compute_weighted_score, axis=1)

    if diabetes_probability >= 0.5:
        sorted_df = df.sort_values(by="weighted_score", ascending=False)

        # filter out recipes with no diabetic-friendly label
        top_friendly = sorted_df[sorted_df["is_diabetic_friendly"] == 1].head(3).copy()
        top_friendly["diabetic_note"] = "suitable"

        # filter out recipes with diabetic-friendly label
        top_unfriendly = sorted_df[sorted_df["is_diabetic_friendly"] == 0].head(2).copy()
        top_unfriendly["diabetic_note"] = "not suitable"

        # Combine the two DataFrames
        top_recipes = pd.concat([top_friendly, top_unfriendly])
    else:
        # if the user is not diabetic, just recommend the top 5 recipes
        top_recipes = df.sort_values(by="weighted_score", ascending=False).head(5).copy()
        top_recipes["diabetic_note"] = df["is_diabetic_friendly"].replace({1: "suitable", 0: "not suitable"})

    return(top_recipes[["recipeName", "weighted_score", "diabetic_note"]])

# Test the function with example data
if __name__ == "__main__":
    category_scores = {
        "oil_fat": 2,
        "dairy": -3,
        "fruit": 4,
        "carbohydrate": -1,
        "sweetener": 0,
        "processed": -2,
        "legume": 3,
        "vegetable": 5,
        "leafy_vegetable": 4,
        "seafood": -1,
        "high_protein": 2,
        "herb_spice": 1,
        "meat": -4
    }
    diabetes_probability = 0.6

    recommended_recipes = recommend_recipes_for_user(category_scores, diabetes_probability)
    print(recommended_recipes)