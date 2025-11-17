# src/predict.py
import mlflow
import pickle
import os

# Point to DagsHub MLflow
mlflow.set_tracking_uri(" https://github.com/nawazpatana/NewLab3.git")

# Load best run (replace with actual run_id from UI)
BEST_RUN_ID = "your-best-run-id"  # ← Get this from DagsHub MLflow UI

# Download model and vectorizer
logged_model = f'runs:/{BEST_RUN_ID}/model'
model = mlflow.sklearn.load_model(logged_model)

# Download vectorizer artifact
client = mlflow.tracking.MlflowClient()
local_dir = f"./models/{BEST_RUN_ID}"
client.download_artifacts(BEST_RUN_ID, "vectorizer.pkl", local_dir)
with open(f"{local_dir}/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Predict function
def predict_spam(message: str) -> str:
    X = vectorizer.transform([message])
    pred = model.predict(X)[0]
    return "spam" if pred == 1 else "ham"

# Test
print(predict_spam("Free entry in 2 a weekly comp..."))  # Should say "spam"
print(predict_spam("Hey, are we still meeting tomorrow?"))  # Should say "ham"