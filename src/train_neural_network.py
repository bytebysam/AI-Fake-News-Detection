import os
import joblib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
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

print("Training Neural Network model...")

model = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42,
    early_stopping=True,
    verbose=True
)

model.fit(X_train, y_train)

print("Making predictions...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nNeural Network Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Fake News", "Real News"],
)

plt.title("Neural Network Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "neural_network_confusion_matrix.png")
)
plt.close()

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "neural_network.joblib")
)

print("\nNeural Network training completed!")
print("Model and confusion matrix saved.")