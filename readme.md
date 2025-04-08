# Dietary Recommendation Assistant for Diabetes Prevention

An end-to-end AI system that predicts diabetes risk using machine learning and recommends personalized recipes based on user preferences extracted from natural language input. The system combines traditional ML, LLMs (Large Language Models), and structured prompt engineering (via MCP) to deliver health-aware, user-friendly food recommendations.

---

## How to Run

```bash
python main.py
```

## Project Structure

```graphql
├── chatbot/                         # LLM-related logic and chatbot integration
│   ├── chatbot_ui.py                # Optional GUI or CLI interface for chatbot
│   ├── llm_inference.py             # Calls LLM to extract preference scores
│   ├── prompt_template.py           # MCP prompt template for structured input
│   └── notebooks/
│       └── deepseek.ipynb           # Experimental notebook for LLM exploration
│
├── core/                            # Machine learning & prediction logic
│   ├── data/
│   │   └── diabetes.csv             # Input dataset for ML prediction
│   ├── model/                       # Saved ML models and scalers
│   ├── utils/                       # Data cleaning and correlation
│   ├── demo.ipynb                   # ML training, evaluation, visualization
│   └── induce.py                    # Induce models to a .py file
│
├── MCP/                             # Model Context Protocol and recommendation logic
│   ├── data/
│   │   └── diabetic_recipes_with_categories.csv  # Recipe database with tags
│   ├── dev/                         # dev experiments or logs
│   └── utils/
│       ├── ParseJSON.py             # Extracts structured output from LLM JSON
│       └── recipe_recommender.py    # Computes weighted scores & gives recommendations
│
├── main.py                          # Application Enterance
```

### Machine Learning Module

- Data preprocessing: Outlier removal, correlation filtering
- Model training: SVM, RF, MLP, Logistic Regression, etc.
- Soft voting & stacking: For robust ensemble performance
- Metrics tracked: AUC, AUPR, F1, Precision, Recall

### LLM + MCP (Model Context Protocol)

- LLM-based user preference parsing from text (e.g., "I love seafood, avoid sweets")
- Structured scoring: Converts natural language to category scores (–5 to +5)
- MCP ensures consistent, machine-usable output for downstream use

### Recipe Recommendation Logic

- Combines LLM output + ML diabetic risk to score and filter recipes
- Returns top suggestions with “suitable” or “not suitable” diabetic tags
- Final output includes reasoning and explanation

### GUI (Optional Extension)

An easy-to-use interface for non-technical users to enter data, view predictions, and receive recommendations.

## Highlights

- MCP-powered prompt engineering for structured generative AI
- Soft-voting and stacking ensemble models for robust health prediction
- Real-time food recommendation based on health status and food intent
- Clean modular codebase for easy adaptation and extension

## Project Info

- Course: CSYE 7380 - Theory & Prac App AI Gen Model
- Team: Jiaxian Li, evelyn-kk
- Date: April 21, 2025
- Project Title: DiabetaRec: A Personalized, Diabetes-Aware Recipe Recommendation System using Machine Learning and Generative AI
