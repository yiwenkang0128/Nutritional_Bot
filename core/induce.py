import joblib
import numpy as np
import pandas as pd

def predict(record: np.array, model_list: list = ['svm', 'rf', 'mlp', 'logistic', 'nb', 'dt', 'knn'], scaler_name='minMax'):
    if record.ndim == 1:
        record = record.reshape(1, -1)

    # Load the saved scaler
    scaler = joblib.load(f'core/model/{scaler_name}.pkl')
    record_scaled = scaler.transform(record)

    probability_list = []
    results = {}

    # Loop through models
    for model_key in model_list:
        model = joblib.load(f'core/model/{model_key}.pkl')
        prob = model.predict_proba(record_scaled)[:, 1][0]
        results[model_key] = prob
        probability_list.append(prob)

    # Soft voting result (average)
    soft_vote = np.mean(probability_list)
    results['soft_vote'] = soft_vote
    results['predicted_label'] = int(soft_vote >= 0.5)

    return results

if __name__ == "__main__":
    model_list = ['svm', 'rf', 'mlp', 'logistic', 'nb', 'dt', 'knn', 'combiner_final']
    scaler_name = 'minMax'

    # Load one sample (same features as training)
    raw_data = pd.read_csv("core/data/diabetes.csv")
    selected_features = ['Pregnancies', 'Glucose', 'BMI', 'Age']
    sample = raw_data[selected_features].values[0]  # take first row

    prediction_result = predict(sample)

    for k, v in prediction_result.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(prediction_result['predicted_label'])
