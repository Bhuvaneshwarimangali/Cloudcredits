"""
Customer Churn Predictor (Machine Learning + LLM Explanation Layer)
--------------------------------------------------------------------
Pipeline:
  1. Load data (Telco Churn CSV if available, else auto-generate a realistic
     synthetic dataset so the project runs immediately).
  2. Preprocess (missing values, encode categoricals, scale numerics).
  3. Train/test split.
  4. Train a RandomForestClassifier.
  5. Evaluate (accuracy, confusion matrix, classification report).
  6. Plot feature importances.
  7. LLM layer: for any customer, generate a plain-English explanation of
     WHY the model predicts churn/no-churn, using the Anthropic API.

HOW TO GET THE REAL DATASET (recommended, takes 2 minutes):
  1. Go to Kaggle -> search "Telco Customer Churn" (by blastchar).
  2. Download WA_Fn-UseC_-Telco-Customer-Churn.csv
  3. Place it in this folder as: telco_churn.csv
  4. Re-run this script -> it will automatically detect and use it instead
     of synthetic data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

RANDOM_STATE = 42
DATA_PATH = "telco_churn.csv"


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data(path=DATA_PATH, n_synthetic=2000):
    if os.path.exists(path):
        print(f"[INFO] Loading real dataset from {path}")
        df = pd.read_csv(path)
        # Telco dataset quirks: TotalCharges has blank strings, drop customerID
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        return df, True

    print("[INFO] telco_churn.csv not found -> generating synthetic dataset "
          "so you can run the full pipeline right now.")
    rng = np.random.default_rng(RANDOM_STATE)

    tenure = rng.integers(0, 72, n_synthetic)
    monthly_charges = rng.normal(65, 25, n_synthetic).clip(18, 120)
    contract = rng.choice(["Month-to-month", "One year", "Two year"],
                           n_synthetic, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_synthetic, p=[0.35, 0.45, 0.20])
    tech_support = rng.choice(["Yes", "No"], n_synthetic, p=[0.35, 0.65])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_synthetic,
    )
    senior = rng.choice([0, 1], n_synthetic, p=[0.85, 0.15])
    partner = rng.choice(["Yes", "No"], n_synthetic)
    total_charges = monthly_charges * tenure + rng.normal(0, 50, n_synthetic)

    # Build churn probability from a hidden "true" rule + noise, so the model
    # has real signal to learn (mirrors real-world churn drivers).
    churn_score = (
        (contract == "Month-to-month").astype(int) * 0.35
        + (internet == "Fiber optic").astype(int) * 0.2
        + (tech_support == "No").astype(int) * 0.15
        + (payment == "Electronic check").astype(int) * 0.1
        + (tenure < 12).astype(int) * 0.25
        - (tenure > 48).astype(int) * 0.2
        + rng.normal(0, 0.15, n_synthetic)
    )
    churn = (churn_score > np.median(churn_score)).astype(int)
    churn_labels = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "SeniorCitizen": senior,
        "Partner": partner,
        "tenure": tenure,
        "Contract": contract,
        "InternetService": internet,
        "TechSupport": tech_support,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_labels,
    })
    return df, False


# --------------------------------------------------------------------------
# 2. PREPROCESS
# --------------------------------------------------------------------------
def preprocess(df):
    df = df.copy()
    df = df.dropna(subset=["Churn"])
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    target = df["Churn"].map({"Yes": 1, "No": 0})
    features = df.drop(columns=["Churn"])

    encoders = {}
    for col in features.select_dtypes(include="object").columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        encoders[col] = le

    for col in features.select_dtypes(include=[np.number]).columns:
        features[col] = features[col].fillna(features[col].median())

    return features, target, encoders


# --------------------------------------------------------------------------
# 3-5. TRAIN + EVALUATE
# --------------------------------------------------------------------------
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    print("\n=== MODEL EVALUATION ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"ROC-AUC : {roc_auc_score(y_test, probs):.3f}")
    print("\nClassification report:\n", classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("[Saved] confusion_matrix.png")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="teal")
    plt.title("Feature Importance - Churn Drivers")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("[Saved] feature_importance.png")

    return model, scaler, X_test, X_test_scaled, y_test, preds, probs


# --------------------------------------------------------------------------
# 6. LLM EXPLANATION LAYER
# --------------------------------------------------------------------------
def explain_prediction_with_llm(customer_row: pd.Series, prediction: str, probability: float):
    """
    Calls the Anthropic API to turn a raw prediction into a plain-English,
    business-friendly explanation. Requires ANTHROPIC_API_KEY env var.
    Install: pip install anthropic
    """
    try:
        import anthropic
    except ImportError:
        return ("[anthropic package not installed] Run: pip install anthropic\n"
                "Then set ANTHROPIC_API_KEY and re-run this function.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[No ANTHROPIC_API_KEY set] Skipping live LLM call — showing prompt only:\n" + \
               build_explanation_prompt(customer_row, prediction, probability)

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_explanation_prompt(customer_row, prediction, probability)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def build_explanation_prompt(customer_row, prediction, probability):
    details = "\n".join(f"- {k}: {v}" for k, v in customer_row.items())
    return f"""You are a retention analyst. A machine learning model predicts this
customer's churn status as: {prediction} (probability of churn: {probability:.2f}).

Customer profile:
{details}

In 3-4 sentences, explain in plain English why this customer is likely (or
unlikely) to churn based on their profile, and suggest ONE concrete retention
action the company could take."""


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    df, is_real = load_data()
    print(f"[INFO] Dataset shape: {df.shape} | Using real Kaggle data: {is_real}")

    X, y, encoders = preprocess(df)
    model, scaler, X_test, X_test_scaled, y_test, preds, probs = train_and_evaluate(X, y)

    # --- Demo: explain one customer from the test set ---
    idx = 0
    sample_customer = X_test.iloc[idx]
    sample_pred = "Yes" if preds[idx] == 1 else "No"
    sample_prob = probs[idx]

    print("\n=== LLM EXPLANATION DEMO ===")
    explanation = explain_prediction_with_llm(sample_customer, sample_pred, sample_prob)
    print(explanation)
