# 🚀 Customer Churn Prediction using Machine Learning

## 📖 Project Overview

Customer churn prediction is an important business problem in the telecom industry. Identifying customers who are likely to discontinue a service enables companies to improve customer retention and reduce revenue loss.

This project builds a **Machine Learning model** using the **Random Forest Classifier** to predict customer churn based on customer information. It also includes an **optional AI-powered explanation module** that uses the Anthropic Claude API to generate human-readable explanations for model predictions.

This project was completed as part of my **Data Science Internship at CloudCredits Technologies**.

---

## 🎯 Objectives

- Predict whether a customer is likely to churn.
- Analyze the factors influencing customer churn.
- Evaluate model performance using industry-standard metrics.
- Visualize feature importance and model performance.
- Generate AI-powered explanations for predictions (optional).

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Random Forest Classifier
- Anthropic Claude API (Optional)

---

## 📂 Project Structure

```
Customer-Churn-Predictor/
│
├── churn_predictor.py
├── requirements.txt
├── README.md
├── confusion_matrix.png
├── feature_importance.png
```

---

## ⚙️ Project Workflow

1. Load customer dataset.
2. Perform data preprocessing.
3. Encode categorical variables.
4. Split the dataset into training and testing sets.
5. Train a Random Forest Classifier.
6. Evaluate model performance.
7. Generate visualizations.
8. Generate AI explanation (optional).

---

## 🤖 Machine Learning Model

**Algorithm Used**

- Random Forest Classifier

---

## 📊 Model Performance

The project produced the following results:

| Metric | Score |
|--------|-------|
| Accuracy | **84%** |
| ROC-AUC | **93%** |

The model also generates:

- Classification Report
- Confusion Matrix
- Feature Importance Graph

---

## 📷 Project Output

After successful execution, the project generates:

- `confusion_matrix.png`
- `feature_importance.png`

These outputs help visualize the model's performance and identify the most influential features affecting customer churn.

---

## 🧠 AI Explanation Module

This project includes an optional AI explanation feature using the **Anthropic Claude API**.

When an API key is configured, the model generates a business-friendly explanation that describes:

- Why a customer is likely or unlikely to churn.
- Key factors affecting the prediction.
- A suggested customer retention strategy.

If no API key is configured, the Machine Learning model still runs successfully, and the project displays the prompt that would be sent to the AI model.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

Navigate to the project folder:

```bash
cd customer-churn-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python churn_predictor.py
```

---


## 🚀 Future Enhancements

- Streamlit Web Application
- Hyperparameter Tuning
- Explainable AI using SHAP
- REST API Deployment
- Docker Support
- Real-time Customer Prediction Dashboard

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Data preprocessing
- Feature engineering
- Machine Learning model development
- Model evaluation
- Data visualization
- AI-assisted prediction explanation

---

## 👩‍💻 Author
## 👩‍💻 Author

**Bhuvaneshwari Mangali**

B.Tech (Computer Science Engineering)

Data Science Intern | CloudCredits Technologies

### Connect with Me

- GitHub: https://github.com/Bhuvaneshwarimangali
- LinkedIn: https://www.linkedin.com/in/bhuvaneshwari-mangali-5a525632a

If you found this project useful, feel free to ⭐ the repository.
