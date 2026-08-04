# Student Health Risk Predictor

A machine learning web app that predicts a student's health risk category (fit, unhealthy, or at-risk) based on lifestyle and physiological data like sleep duration, stress level, activity level, and more.

Built as part of the CIS6005 Computational Intelligence mini project at Cardiff Metropolitan University.

**Live app:** https://student-health-risk-app-njq4rqkcdfl3e3sh7c946g.streamlit.app/

## What it does

You fill in a short form with lifestyle info (sleep hours, stress level, exercise, diet, etc.) and the app predicts which health risk category you're likely to fall into, along with a probability breakdown for each category so it's clear how confident the model actually is, rather than just spitting out a single label.

## The model

The dataset came from a Kaggle competition ("Predicting Student Health Risk") with about 690,000 rows and 14 features. Before landing on a final model, I compared four different approaches on the same data:

| Model | Accuracy |
|---|---|
| **Gradient Boosting** | **96.51%** |
| Random Forest | 96.49% |
| Decision Tree | 94.15% |
| Logistic Regression | 68.20% |

Gradient Boosting came out on top (just barely ahead of Random Forest) so that's the model deployed here.

The two strongest predictors by a wide margin were `sleep_duration` and `stress_level` — between them they accounted for roughly 59% of the model's feature importance.

## How the app works

1. You fill in the form (sleep, stress, activity level, BMI, diet, etc.)
2. The app encodes your categorical answers using the same `LabelEncoder`s the model was trained on
3. The row gets reordered to match the exact column order the model expects
4. The model runs `.predict()` and `.predict_proba()` on your input
5. The result is shown with a colour-coded box and an expandable probability breakdown

## Running it locally

Clone the repo and install the requirements:

```bash
git clone <this-repo-url>
cd student_health_app
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Project structure

```
student_health_app/
├── app.py              # Streamlit app — form, prediction logic, results display
├── gb_model.pkl         # Trained Gradient Boosting model
├── encoders.pkl          # LabelEncoders for categorical columns
├── columns.pkl            # Column order the model expects
├── requirements.txt        # Python dependencies
└── runtime.txt              # Python version for Streamlit Cloud
```

## Tech stack

- **Python 3.11**
- **scikit-learn** (1.6.1) — model training and inference
- **Streamlit** — web app frontend
- **pandas** — data handling

## Notes

- This is a lifestyle-based risk estimate, not a medical diagnosis. The probability breakdown is shown deliberately instead of a single flat label, to communicate that this is a rough, data-driven estimate rather than a verdict.
- The dataset is heavily imbalanced (about 86% of rows are labelled "at-risk"), so accuracy alone isn't the full picture — recall on the minority classes (fit / unhealthy) was checked separately during model evaluation.
