# Customer Churn Predictor

AI Lab Project — a Streamlit app that trains and compares machine learning
models to predict telecom customer churn, with a visual, explainable UI.

**AI approach used:** Option 3 — Machine Learning AI (classification).

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How to use it

1. **Problem Setup** — use the bundled sample dataset (`data/churn_data.csv`)
   or upload your own CSV with the same columns.
2. **Train Models** — pick two algorithms (Random Forest and/or Logistic
   Regression) and hyperparameters, then click **Run training**.
3. **Visual Results** — confusion matrix, ROC curve, and feature importance
   chart for each model.
4. **Explainability** — pick any test customer and see a plain-language
   breakdown of what drove their prediction.
5. **Evaluation** — side-by-side metrics table (accuracy, precision, recall,
   F1, ROC AUC) and a comparison chart, with the better model called out.

## Project structure

```
ChurnPredictor_HasnainM/
├── app.py              # Streamlit UI (render_ui + one function per tab)
├── logic.py             # Core logic: load_data, preprocess_data,
│                         # run_model_or_algorithm, generate_explanation
├── requirements.txt
├── README.md
├── report.md            # Short report (problem, method, AI used, results)
├── data/
│   ├── churn_data.csv   # Sample dataset (synthetic, 800 rows)
│   └── make_dataset.py  # Script that generated churn_data.csv
└── screenshots/         # Add screenshots of the running app here
```

## Dataset

`data/churn_data.csv` is a synthetically generated but realistic telecom
churn dataset (800 customers, ~29% churn rate). Features: tenure, monthly
and total charges, contract type, internet service, tech support, number of
support calls, and payment method. It was built with a hand-designed logit
model (see `data/make_dataset.py`) so the churn label reflects believable
real-world relationships (e.g. month-to-month contracts and high support-call
counts increase churn risk; longer tenure decreases it).

You can also upload your own CSV as long as it has these columns:
`tenure_months, monthly_charges, total_charges, contract_type,
internet_service, tech_support, num_support_calls, payment_method, churn`.

## Notes on AI integration

No external AI API is used — both models (Random Forest, Logistic
Regression) are trained locally with scikit-learn on the loaded dataset, so
there's nothing to disclose re: external prompts/inputs/outputs/cost.
