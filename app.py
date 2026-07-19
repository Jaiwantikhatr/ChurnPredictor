"""
Customer Churn Predictor — AI Lab Project
Option 3 (Machine Learning AI): train a classifier on a dataset, show
training results and predictions visually.

Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

from logic import (
    load_data, preprocess_data, run_model_or_algorithm,
    generate_explanation, get_feature_importance,
)

SAMPLE_DATA_PATH = "data/churn_data.csv"

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "df": None, "prepped": None,
        "result_a": None, "result_b": None,
        "trained": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# A) Problem Setup Module
# ---------------------------------------------------------------------------
def render_problem_setup():
    st.header("1️⃣ Problem Setup")
    st.markdown(
        "**Problem:** predict whether a telecom customer will churn (cancel service), "
        "given their account and usage details.\n\n"
        "- **Input:** customer features (tenure, charges, contract type, support usage, etc.)\n"
        "- **Output:** churn probability + Yes/No prediction\n"
        "- **Constraint:** binary classification; needs a `churn` column (0/1) in the data"
    )

    source = st.radio(
        "Choose your data source", ["Use sample dataset", "Upload my own CSV"], horizontal=True
    )

    if source == "Use sample dataset":
        try:
            df = load_data(SAMPLE_DATA_PATH)
            st.success(f"✅ Loaded sample dataset: {len(df)} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Could not load sample dataset: {e}")
            return
    else:
        uploaded = st.file_uploader("Upload a churn CSV", type=["csv"])
        if uploaded is None:
            st.info("Waiting for a CSV upload...")
            return
        try:
            df = load_data(uploaded)
            st.success(f"✅ File validated and loaded: {len(df)} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Invalid file: {e}")
            return

    st.session_state["df"] = df
    with st.expander("Preview data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Churn rate", f"{df['churn'].mean():.1%}")
    col3.metric("Features", df.shape[1] - 1)


# ---------------------------------------------------------------------------
# B) Core Logic Module — training controls
# ---------------------------------------------------------------------------
def render_training():
    st.header("2️⃣ Train & Compare Models")

    if st.session_state["df"] is None:
        st.warning("⚠️ Load data in step 1 first.")
        return

    df = st.session_state["df"]

    st.markdown("Configure two models to compare side-by-side (required: at least two approaches).")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Model A")
        model_a_name = st.selectbox("Algorithm A", ["Random Forest", "Logistic Regression"], key="ma")
        params_a = {}
        if model_a_name == "Random Forest":
            params_a["n_estimators"] = st.slider("n_estimators (A)", 50, 400, 150, 25)
            params_a["max_depth"] = st.slider("max_depth (A)", 2, 20, 6)
            params_a["min_samples_leaf"] = st.slider("min_samples_leaf (A)", 1, 10, 3)
        else:
            params_a["C"] = st.slider("Regularization C (A)", 0.01, 5.0, 1.0)

    with c2:
        st.subheader("Model B")
        model_b_name = st.selectbox("Algorithm B", ["Logistic Regression", "Random Forest"], key="mb")
        params_b = {}
        if model_b_name == "Random Forest":
            params_b["n_estimators"] = st.slider("n_estimators (B)", 50, 400, 150, 25)
            params_b["max_depth"] = st.slider("max_depth (B)", 2, 20, 6)
            params_b["min_samples_leaf"] = st.slider("min_samples_leaf (B)", 1, 10, 3)
        else:
            params_b["C"] = st.slider("Regularization C (B)", 0.01, 5.0, 1.0)

    if st.button("🚀 Run training", type="primary"):
        status = st.empty()
        try:
            status.info("Preprocessing data...")
            prepped = preprocess_data(df)
            time.sleep(0.2)

            status.info(f"Training {model_a_name} (Model A)...")
            result_a = run_model_or_algorithm(prepped, model_a_name, params_a)
            time.sleep(0.2)

            status.info(f"Training {model_b_name} (Model B)...")
            result_b = run_model_or_algorithm(prepped, model_b_name, params_b)

            st.session_state["prepped"] = prepped
            st.session_state["result_a"] = result_a
            st.session_state["result_b"] = result_b
            st.session_state["trained"] = True
            status.success("✅ Training complete — see Results, Explainability, and Evaluation tabs.")
        except Exception as e:
            status.error(f"❌ Training failed: {e}")


# ---------------------------------------------------------------------------
# C) Visual UI Module — results & charts
# ---------------------------------------------------------------------------
def render_results():
    st.header("3️⃣ Visual Results")

    if not st.session_state["trained"]:
        st.warning("⚠️ Train models in step 2 first.")
        return

    prepped = st.session_state["prepped"]
    for label, result in [("Model A", st.session_state["result_a"]), ("Model B", st.session_state["result_b"])]:
        st.subheader(f"{label}: {result['model_name']}")
        cols = st.columns([1, 1, 1])

        # Confusion matrix heatmap
        cm = result["confusion_matrix"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No churn", "Churn"], y=["No churn", "Churn"],
        )
        fig_cm.update_layout(title="Confusion Matrix", height=320, margin=dict(t=40, b=10))
        cols[0].plotly_chart(fig_cm, use_container_width=True)

        # ROC curve
        fpr, tpr = result["roc_curve"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig_roc.update_layout(
            title=f"ROC Curve (AUC={result['metrics']['roc_auc']:.2f})",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=320, margin=dict(t=40, b=10),
        )
        cols[1].plotly_chart(fig_roc, use_container_width=True)

        # Feature importance
        pairs = get_feature_importance(result, prepped["feature_names"])[:8]
        feats, vals = zip(*pairs)
        fig_imp = px.bar(
            x=list(vals)[::-1], y=[f.replace("_", " ") for f in feats][::-1],
            orientation="h", labels={"x": "Importance", "y": ""},
        )
        fig_imp.update_layout(title="Top Feature Importances", height=320, margin=dict(t=40, b=10))
        cols[2].plotly_chart(fig_imp, use_container_width=True)


# ---------------------------------------------------------------------------
# D) Explainability Module
# ---------------------------------------------------------------------------
def render_explainability():
    st.header("4️⃣ Explainability")

    if not st.session_state["trained"]:
        st.warning("⚠️ Train models in step 2 first.")
        return

    prepped = st.session_state["prepped"]
    which = st.radio("Explain a prediction from:", ["Model A", "Model B"], horizontal=True)
    result = st.session_state["result_a"] if which == "Model A" else st.session_state["result_b"]

    n_test = len(prepped["X_test"])
    idx = st.slider("Pick a test customer (row index)", 0, n_test - 1, 0)

    st.dataframe(prepped["X_test_raw"].iloc[[idx]], use_container_width=True)

    explanation_text, top_contribs = generate_explanation(
        result, prepped["feature_names"], idx, prepped
    )

    st.markdown("### Result panel")
    proba = result["y_proba"][idx]
    st.metric("Churn probability", f"{proba:.0%}")
    st.progress(min(max(proba, 0.0), 1.0))
    st.markdown(explanation_text)

    feats = [c[0].replace("_", " ") for c in top_contribs]
    contribs = [c[1] for c in top_contribs]
    fig = px.bar(
        x=contribs, y=feats, orientation="h",
        color=[("increases risk" if c > 0 else "decreases risk") for c in contribs],
        color_discrete_map={"increases risk": "#d62728", "decreases risk": "#2ca02c"},
        labels={"x": "Contribution", "y": ""},
    )
    fig.update_layout(title="Why this prediction? (top contributing factors)", height=300)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# E) Evaluation Module
# ---------------------------------------------------------------------------
def render_evaluation():
    st.header("5️⃣ Evaluation & Comparison")

    if not st.session_state["trained"]:
        st.warning("⚠️ Train models in step 2 first.")
        return

    result_a = st.session_state["result_a"]
    result_b = st.session_state["result_b"]

    rows = []
    for label, r in [("Model A: " + result_a["model_name"], result_a), ("Model B: " + result_b["model_name"], result_b)]:
        m = r["metrics"]
        rows.append({
            "Model": label, "Accuracy": m["accuracy"], "Precision": m["precision"],
            "Recall": m["recall"], "F1 score": m["f1"], "ROC AUC": m["roc_auc"],
        })
    metrics_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(metrics_df.style.format("{:.3f}").highlight_max(axis=0, color="#c6f6d5"), use_container_width=True)

    melted = metrics_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(melted, x="Metric", y="Value", color="Model", barmode="group")
    fig.update_layout(title="Model comparison across metrics", height=380)
    st.plotly_chart(fig, use_container_width=True)

    better = metrics_df["F1 score"].idxmax()
    st.success(f"🏆 By F1 score, **{better}** performs better on this test split.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def render_ui():
    init_state()
    st.title("📉 Customer Churn Predictor")
    st.caption("AI Lab Project — Machine Learning classifier with explainability, built in Streamlit")

    tabs = st.tabs([
        "1. Problem Setup", "2. Train Models", "3. Visual Results",
        "4. Explainability", "5. Evaluation",
    ])
    with tabs[0]:
        render_problem_setup()
    with tabs[1]:
        render_training()
    with tabs[2]:
        render_results()
    with tabs[3]:
        render_explainability()
    with tabs[4]:
        render_evaluation()


if __name__ == "__main__":
    render_ui()
