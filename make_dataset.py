"""
Generates a synthetic (but realistic) telecom customer churn dataset.
Run once to produce data/churn_data.csv. Not needed at app runtime since
the CSV is already committed, but kept here so graders can see how the
sample data was built (per submission requirement #4).
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 800

tenure_months = np.random.randint(0, 72, N)
monthly_charges = np.round(np.random.uniform(18, 120, N), 2)
contract_type = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20]
)
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.20])
tech_support = np.random.choice(["Yes", "No"], N, p=[0.4, 0.6])
num_support_calls = np.random.poisson(1.5, N)
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N
)

total_charges = np.round(monthly_charges * np.maximum(tenure_months, 1) * np.random.uniform(0.9, 1.0, N), 2)

# Build churn probability from a hand-designed logit so the model has real signal to learn
logit = (
    -3.0
    + 0.025 * monthly_charges
    - 0.045 * tenure_months
    + 0.28 * num_support_calls
    + np.where(contract_type == "Month-to-month", 1.1, 0)
    + np.where(contract_type == "One year", 0.2, 0)
    + np.where(internet_service == "Fiber optic", 0.5, 0)
    + np.where(tech_support == "No", 0.5, 0)
    + np.where(payment_method == "Electronic check", 0.35, 0)
)
prob = 1 / (1 + np.exp(-logit))
churn = np.random.binomial(1, prob)

df = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "contract_type": contract_type,
    "internet_service": internet_service,
    "tech_support": tech_support,
    "num_support_calls": num_support_calls,
    "payment_method": payment_method,
    "churn": churn,
})

df.to_csv("churn_data.csv", index=False)
print(df["churn"].value_counts(normalize=True))
print("Saved churn_data.csv with", len(df), "rows")
