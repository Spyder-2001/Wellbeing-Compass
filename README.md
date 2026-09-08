# 🧭 Wellbeing Compass

**An AI-Powered Mental Wellbeing Assistant for Personalized Risk Assessment and Recommendation**

Course: **CSE445 – Machine Learning**
North South University

## 👥 Team — Group 7
- Tajkira Kalim Nity
- Md. Shafayat Kabir Bhuyan
- Sazzad Hossain Shawon

---

## 📌 Overview

Wellbeing Compass is an AI-powered mental wellbeing assistant that combines demographic, occupational, and lifestyle indicators to generate a personalized wellbeing assessment and produce tailored, non-clinical recommendations. The system is designed as an **early-awareness and self-care companion** — not a diagnostic instrument — and routes users toward professional care when indicators suggest elevated risk.

## 🎯 Problem Statement

Mental health conditions such as depression, anxiety, and chronic stress are among the leading contributors to the global burden of disease. Despite this, significant diagnosis and treatment gaps persist due to limited clinic hours, geographic inaccessibility, workforce shortages, and social stigma. Wellbeing Compass aims to provide a low-friction, privacy-respecting tool that helps individuals notice worsening stress, sleep, or mood patterns before they escalate.

## ✨ Features

- **Multi-factor risk assessment** using demographic (age, gender, occupation, country), lifestyle (sleep, work hours, physical activity), and self-reported (stress level, consultation history) attributes
- **Binary mental-health-condition classification** (at-risk / not at-risk)
- **Four-level severity classification** (None / Low / Medium / High)
- **Engineered risk features**: Sleep Deficit, Overtime Hours, Work-Sleep Ratio, Low Activity flag, and a composite Lifestyle Risk Score
- **Personalized, actionable recommendations** (sleep hygiene, activity suggestions, stress management, professional referral prompts)
- **Interactive Streamlit web app** with a wellbeing "compass" dashboard and risk factor visualization

## 🧠 Methodology

The end-to-end pipeline follows five stages:
1. **Requirement Analysis** — define objectives and key features
2. **Data Collection & Preprocessing** — collect and clean demographic, lifestyle, and health-history attributes
3. **Model Development & Training** — train and tune ML classifiers for risk and severity classification
4. **System Development & Integration** — build the application and user interface
5. **Testing, Deployment & Evaluation** — evaluate performance and deploy

### Dataset
- 1,000 users, 12 attributes (Age, Gender, Occupation, Country, Sleep Hours, Work Hours, Physical Activity, Stress Level, Consultation History, Mental Health Condition, Severity)
- Balanced target classes (515 positive / 485 negative)

### Models
Four supervised classifiers were trained and compared for binary classification, with hyperparameters tuned via `GridSearchCV`:
- Logistic Regression
- Gradient Boosting
- Support Vector Machine (RBF kernel)
- Random Forest

A separate multi-class classifier was trained for four-level severity classification.

## 📊 Results

| Model | Accuracy | Balanced Acc. | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.490 | 0.4898 | 0.5050 | 0.4951 | 0.5000 |
| Gradient Boosting | 0.490 | 0.4880 | 0.5044 | 0.5534 | 0.5278 |
| SVM (RBF) | 0.495 | 0.4953 | 0.5102 | 0.4854 | 0.4975 |
| Random Forest | 0.445 | 0.4416 | 0.4672 | 0.5534 | 0.5067 |

Severity (4-class) classifier: **47.0% accuracy**, **23.5% balanced accuracy**.

> These results, close to chance level, indicate that purely structured demographic/lifestyle attributes carry limited discriminative signal for this dataset — motivating future incorporation of free-text/journaling signals processed with a compact language model.

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn, imbalanced-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Experimentation | Google Colab / Kaggle |
| Model Hub | Hugging Face |

## 📂 Project Structure

```
├── Wellbeing_Compass.ipynb      # Full training pipeline (EDA, feature engineering, model training)
├── web.py                       # Streamlit deployment app
├── deployment_artifacts/        # Saved models & preprocessing artifacts
└── README.md
```

## 🚀 Running the App

```bash
pip install -r requirements.txt
streamlit run web.py
```

## 🔮 Future Work

- Incorporate an optional free-text journaling feature processed with a compact on-device language model
- Collect a larger, real-world (anonymized, consented) dataset
- Address class imbalance in severity classification via resampling/focal loss
- Add model explainability (SHAP feature importance)
- Explore self-play synthetic data augmentation for a conversational check-in feature
- Conduct a small-scale user study and clinical review before real-world deployment
- Extend features with wearable-derived signals (heart-rate variability, activity tracking)

## ⚠️ Disclaimer

Wellbeing Compass is a **non-clinical, self-awareness tool** and is not a substitute for professional diagnosis or treatment. If you are experiencing a mental health crisis, please consult a licensed professional or local emergency services.
