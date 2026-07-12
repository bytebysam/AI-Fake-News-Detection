import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading train-test data...")

X_train, X_test, y_train, y_test = joblib.load(
    os.path.join(MODEL_DIR, "train_test_data.joblib")
)

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Making predictions...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nRandom Forest Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Real News", "Fake News"],
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "random_forest_confusion_matrix.png")
)
plt.close()

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "random_forest.joblib")
)

print("\nRandom Forest training completed!")
print("Model and confusion matrix saved.")