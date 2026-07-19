"""
Core logic module — kept separate from the UI (app.py) as required by the
project spec ("Keep code modular (separate logic from UI)").

Function names roughly follow the suggested reference structure:
    load_data, preprocess_data, run_model_or_algorithm,
    generate_explanation, create_visuals (visuals live in app.py since
    they're Plotly figures rendered directly into Streamlit)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

REQUIRED_COLUMNS = [
    "tenure_months", "monthly_charges", "total_charges", "contract_type",
    "internet_service", "tech_support", "num_support_calls",
    "payment_method", "churn",
]

CATEGORICAL_COLS = ["contract_type", "internet_service", "tech_support", "payment_method"]
NUMERIC_COLS = ["tenure_months", "monthly_charges", "total_charges", "num_support_calls"]


def load_data(path_or_buffer):
    """Load a churn CSV and validate it has the columns this app expects."""
    df = pd.read_csv(path_or_buffer)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
        )
    df = df.dropna()
    if df["churn"].nunique() != 2:
        raise ValueError("The 'churn' column must be binary (0/1) for this classifier.")
    return df.reset_index(drop=True)


def preprocess_data(df):
    """
    One-hot encode categoricals, scale numerics, split into train/test.
    Returns everything downstream steps need, plus the fitted scaler and
    the final feature column order (needed later for explanations).
    """
    X = pd.get_dummies(df.drop(columns=["churn"]), columns=CATEGORICAL_COLS, drop_first=True)
    y = df["churn"]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test_scaled[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names, "scaler": scaler,
        "X_test_raw": X_test,  # unscaled, for human-readable explanations
    }


def run_model_or_algorithm(prepped, model_name, params):
    """Train the selected model and return it plus test-set predictions/metrics."""
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 150),
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=params.get("min_samples_leaf", 3),
            random_state=42,
        )
    else:  # Logistic Regression
        model = LogisticRegression(
            C=params.get("C", 1.0), max_iter=1000, random_state=42
        )

    model.fit(prepped["X_train"], prepped["y_train"])

    y_pred = model.predict(prepped["X_test"])
    y_proba = model.predict_proba(prepped["X_test"])[:, 1]

    metrics = {
        "accuracy": accuracy_score(prepped["y_test"], y_pred),
        "precision": precision_score(prepped["y_test"], y_pred, zero_division=0),
        "recall": recall_score(prepped["y_test"], y_pred, zero_division=0),
        "f1": f1_score(prepped["y_test"], y_pred, zero_division=0),
        "roc_auc": roc_auc_score(prepped["y_test"], y_proba),
    }
    cm = confusion_matrix(prepped["y_test"], y_pred)
    fpr, tpr, _ = roc_curve(prepped["y_test"], y_proba)

    return {
        "model": model, "model_name": model_name,
        "y_pred": y_pred, "y_proba": y_proba,
        "metrics": metrics, "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
    }


def get_feature_importance(result, feature_names):
    """Return a sorted (feature, importance) list, works for RF and Logistic Regression alike."""
    model = result["model"]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        vals = np.abs(model.coef_[0])
        vals = vals / vals.sum()  # normalize coefficients so scale matches RF importances
    pairs = sorted(zip(feature_names, vals), key=lambda p: p[1], reverse=True)
    return pairs


def _feature_directions(prepped):
    """
    RF feature_importances_ are unsigned, so on their own they can't tell you
    whether a *high* value pushes risk up or down. We estimate direction from
    how each feature correlates with the training target, then combine that
    sign with the model's importance magnitude for the explanation.
    """
    X_train = prepped["X_train"]
    y_train = prepped["y_train"]
    directions = {}
    for feat in X_train.columns:
        corr = np.corrcoef(X_train[feat], y_train)[0, 1]
        directions[feat] = 1 if (np.isnan(corr) or corr >= 0) else -1
    return directions


def generate_explanation(result, feature_names, sample_idx, prepped):
    """
    Plain-language explanation for a single prediction: which features pushed
    it toward/away from churn, combining the model's importance (magnitude)
    with the feature's real-world direction of effect (sign) and how far
    this customer's value sits from the training average.
    """
    X_test_scaled = prepped["X_test"]
    X_test_raw = prepped["X_test_raw"]

    row_scaled = X_test_scaled.iloc[sample_idx]
    row_raw = X_test_raw.iloc[sample_idx]
    proba = result["y_proba"][sample_idx]
    pred = result["y_pred"][sample_idx]

    importances = dict(get_feature_importance(result, feature_names))
    directions = _feature_directions(prepped)

    contributions = []
    for feat in feature_names:
        weight = importances.get(feat, 0)
        sign = directions.get(feat, 1)
        val = row_scaled[feat]
        # magnitude from importance & how unusual this value is; sign from real-world correlation
        contributions.append((feat, sign * weight * val, row_raw[feat]))

    contributions.sort(key=lambda c: abs(c[1]), reverse=True)
    top = contributions[:4]

    verdict = "likely to churn" if pred == 1 else "likely to stay"
    lines = [f"This customer is predicted **{verdict}** (churn probability: {proba:.0%})."]
    lines.append("Top factors influencing this prediction:")
    for feat, contrib, raw_val in top:
        direction = "increases" if contrib > 0 else "decreases"
        readable = feat.replace("_", " ")
        lines.append(f"- **{readable}** = {raw_val} → {direction} churn risk")

    return "\n".join(lines), top
