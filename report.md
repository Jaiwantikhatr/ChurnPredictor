# Short Report — Customer Churn Predictor

## Problem
Telecom companies lose significant revenue to customer churn. This project
predicts whether a customer will churn based on their account details
(tenure, charges, contract type) and service usage (internet type, tech
support, support-call frequency). Input is a row of customer features;
output is a churn probability and Yes/No prediction, with an explanation of
why.

## Method
- **Data:** 800-row synthetic-but-realistic telecom dataset
  (`data/churn_data.csv`), generated from a hand-designed logistic model so
  the churn label follows believable real-world relationships (month-to-month
  contracts, high monthly charges, and frequent support calls raise churn
  risk; longer tenure lowers it).
- **Preprocessing:** categorical features one-hot encoded, numeric features
  standardized; 75/25 stratified train/test split.
- **Models compared:**
  - Random Forest Classifier (150 trees, max_depth 6, min_samples_leaf 3)
  - Logistic Regression (C=1.0)
- **Explainability:** for a chosen test customer, the app ranks features by
  `importance × real-world direction (from feature–target correlation) ×
  how far the customer's value deviates from average`, then renders this as
  a plain-language explanation and a bar chart.

## AI used
Both models are standard scikit-learn classifiers trained locally at
runtime on the loaded dataset — this is Option 3 (Machine Learning AI) from
the project guide. No external AI API is used.

## Results (default hyperparameters, same train/test split)

| Model               | Accuracy | Precision | Recall | F1    | ROC AUC |
|---------------------|---------:|----------:|-------:|------:|--------:|
| Random Forest        | 0.755    | 0.654     | 0.298  | 0.410 | 0.794   |
| Logistic Regression   | 0.815    | 0.727     | 0.561  | 0.634 | 0.823   |

On this dataset, Logistic Regression outperforms the default Random Forest
across every metric — likely because the underlying churn relationship was
generated from a (roughly) linear logit function, which favors a linear
model. The app lets you re-tune Random Forest's depth/leaf-size/tree-count
interactively, which closes most of the gap.

## Limitations & future improvements
- Dataset is synthetic; results would need validation on real telecom churn
  data before any business use.
- Explanation module is a lightweight importance-and-correlation heuristic,
  not a formal method like SHAP — good enough for classroom explainability,
  but not for production-grade interpretability.
- Could add cross-validation, hyperparameter search, and more model options
  (e.g. Gradient Boosting) in a future iteration.
