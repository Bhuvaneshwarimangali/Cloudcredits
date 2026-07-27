# Customer Churn Predictor (ML + LLM Explanation Layer)

Predicts whether a telecom customer will churn, then uses an LLM to explain
*why* in plain English — turning a black-box classifier into an AI advisor.

## What it demonstrates
- Machine Learning: classification (Random Forest), feature engineering, evaluation metrics
- Applied AI: LLM-generated natural-language explanations of model predictions

## Setup (5 minutes)
```bash
pip install -r requirements.txt
```

Optional (for real dataset instead of synthetic):
1. Kaggle → search "Telco Customer Churn" (by blastchar)
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Rename to `telco_churn.csv`, put in this folder

Optional (for live LLM explanations instead of a printed prompt):
```bash
export ANTHROPIC_API_KEY="your-key-here"
```
If you don't have a key, the script still runs and prints the exact prompt
that would be sent — fine for a demo video, just mention it's the LLM step.

## Run
```bash
python churn_predictor.py
```

## Outputs
- `confusion_matrix.png` — model performance
- `feature_importance.png` — key churn drivers (great for your video)
- Console: accuracy, ROC-AUC, classification report, and an LLM-generated
  explanation for a sample customer

## For your demo video
1. Show the terminal output (accuracy/AUC).
2. Show `feature_importance.png` — explain top 2-3 churn drivers.
3. Show the LLM explanation text — this is your "AI layer" highlight.
4. Mention the pipeline: data → preprocessing → RF model → LLM reasoning.

## Push to GitHub
```bash
git init
git add .
git commit -m "Customer Churn Predictor - Cloudcredits internship task"
git remote add origin <your-Cloudcredits-repo-url>
git push -u origin main
```
