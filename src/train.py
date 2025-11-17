# src/train.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import dagshub

# === 1. Setup DagsHub MLflow ===
dagshub.init(
    repo_owner='nawazishpatana',   # ← CHANGE THIS
    repo_name='NewLab3',      # ← CHANGE THIS
    mlflow=True
)

# This sets remote tracking URI automatically
# You can also manually set:
# mlflow.set_tracking_uri("https://dagshub.com/your-username/spam-mlflow.mlflow")

mlflow.set_experiment("spam_detection")

def load_data():
    df = pd.read_csv("data/spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df.dropna()

def train_model(model_name, model, vectorizer_params, model_params=None):
    with mlflow.start_run(run_name=model_name):
        # Log params
        mlflow.log_params({
            "model": model_name,
            **vectorizer_params,
            **(model_params or {})
        })

        # Load & prepare data
        df = load_data()
        vectorizer = TfidfVectorizer(**vectorizer_params)
        X = vectorizer.fit_transform(df['message'])
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train
        if model_params:
            model.set_params(**model_params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1
        })

        # Log model & vectorizer
        mlflow.sklearn.log_model(model, "model")
        # Optional: save vectorizer as artifact
        import pickle
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact("vectorizer.pkl")

        print(f"✅ {model_name} → Acc: {acc:.4f}, F1: {f1:.4f}")
        return acc, f1

if __name__ == "__main__":
    experiments = [
        {
            "name": "RF_50",
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "vec_params": {"max_features": 1000, "stop_words": "english"},
            "model_params": {"n_estimators": 50}
        },
        {
            "name": "LR_Basic",
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "vec_params": {"max_features": 1000, "stop_words": "english"},
            "model_params": {}
        },
        {
            "name": "RF_100_HighDim",
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "vec_params": {"max_features": 3000, "stop_words": "english"},
            "model_params": {"n_estimators": 100}
        }
    ]

    results = []
    for exp in experiments:
        acc, f1 = train_model(
            exp["name"],
            exp["model"],
            exp["vec_params"],
            exp["model_params"]
        )
        results.append((exp["name"], acc, f1))
    
    # Find best by F1
    best = max(results, key=lambda x: x[2])
    print(f"\n🏆 Best model: {best[0]} (F1: {best[2]:.4f})")