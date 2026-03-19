"""
Synthetic customer churn dataset generator.

Produces realistic Telco-style churn data with known signal so models
can actually learn something useful.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def generate(n_samples: int = 5_000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a DataFrame of synthetic customer records with a 'Churn' column."""
    rng = np.random.default_rng(random_state)
    random.seed(random_state)

    n = n_samples

    # ── demographic ──────────────────────────────────────────────────────────
    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])

    # ── tenure & contract ────────────────────────────────────────────────────
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]
    )
    # Tenure shaped by contract type
    tenure = np.where(
        contract == "Month-to-month",
        rng.integers(1, 36, n),
        np.where(contract == "One year", rng.integers(12, 60, n), rng.integers(24, 72, n)),
    )

    # ── services ─────────────────────────────────────────────────────────────
    internet_service = rng.choice(["DSL", "Fiber optic", "None"], n, p=[0.34, 0.44, 0.22])
    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    online_security = np.where(
        internet_service == "None",
        "No internet service",
        rng.choice(["Yes", "No"], n, p=[0.30, 0.70]),
    )
    tech_support = np.where(
        internet_service == "None",
        "No internet service",
        rng.choice(["Yes", "No"], n, p=[0.30, 0.70]),
    )
    streaming_tv = np.where(
        internet_service == "None",
        "No internet service",
        rng.choice(["Yes", "No"], n, p=[0.40, 0.60]),
    )

    # ── billing ──────────────────────────────────────────────────────────────
    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.60, 0.40])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges driven by services
    base_charge = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
    monthly_charges = (
        base_charge
        + (phone_service == "Yes").astype(float) * rng.uniform(5, 15, n)
        + (streaming_tv == "Yes").astype(float) * rng.uniform(5, 10, n)
        + rng.normal(0, 3, n)
    ).clip(18, 120).round(2)

    total_charges = (monthly_charges * tenure + rng.normal(0, 20, n)).clip(0).round(2)

    # ── support interactions ─────────────────────────────────────────────────
    # Unhappy customers call more often
    support_calls = rng.poisson(lam=np.where(internet_service == "Fiber optic", 2.5, 1.2), size=n)

    # ── churn label (logistic-style probabilities) ────────────────────────────
    # Strong signals: short tenure, month-to-month, fiber optic, no security,
    # high monthly charges, electronic check, many support calls
    logit = (
        -3.0
        + (-0.06) * tenure
        + (contract == "Month-to-month") * 1.8
        + (contract == "One year") * 0.4
        + (internet_service == "Fiber optic") * 0.7
        + (internet_service == "None") * (-0.8)
        + (online_security == "No") * 0.6
        + (tech_support == "No") * 0.5
        + (paperless_billing == "Yes") * 0.3
        + (payment_method == "Electronic check") * 0.5
        + 0.015 * monthly_charges
        + 0.25 * support_calls
        + senior * 0.3
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob_churn).astype(str)
    churn = np.where(churn == "1", "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:05d}" for i in range(n)],
            "gender": gender,
            "SeniorCitizen": senior,
            "tenure": tenure,
            "PhoneService": phone_service,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "SupportCalls": support_calls,
            "Churn": churn,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn dataset")
    parser.add_argument("--n-samples", type=int, default=5_000)
    parser.add_argument("--out", default="data/raw/churn_data.csv")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(n_samples=args.n_samples, random_state=args.seed)
    df.to_csv(out, index=False)
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Generated {len(df):,} rows → {out}")
    print(f"Churn rate: {churn_rate:.1%}  ({(df['Churn']=='Yes').sum():,} churned)")


if __name__ == "__main__":
    main()
