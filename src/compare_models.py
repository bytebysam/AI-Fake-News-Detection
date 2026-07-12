import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading test data...")

_, X_test, _, y_test = joblib.load(
    os.path.join(MODEL_DIR, "train_test_data.joblib")
)

models = {
    "KNN": "knn.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "Neural Network": "neural_network.joblib",
}

results = []

for name, filename in models.items():
    print(f"Evaluating {name}...")

    model = joblib.load(
        os.path.join(MODEL_DIR, filename)
    )

    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

plt.figure(figsize=(9, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"] * 100
)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_accuracy_comparison.png")
)

plt.close()

print("\nModel comparison completed!")
print("Results and accuracy graph saved.")