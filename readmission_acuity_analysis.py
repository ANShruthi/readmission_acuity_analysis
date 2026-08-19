"""
Real-World Readmission Risk Analytics: Acuity Proxies & Economic
Threshold Sensitivity in Diabetic Inpatients

Author: Shruthi Nagaraju, MD, MHA (DHA Candidate)

Data: UCI "Diabetes 130-US hospitals for years 1999-2008" dataset
(real, publicly available, de-identified inpatient encounters).

NOTE ON COSTS: This dataset does not contain billing/claims cost fields.
The intervention cost/benefit assumptions used in the threshold sensitivity
analysis below (cost of a false-positive outreach call, cost of a missed
readmission, benefit of a prevented readmission) are ILLUSTRATIVE, sourced
from published readmission-cost literature ranges, and explicitly labeled
as assumptions -- not observed data. This mirrors how real health-economics
teams often have to pair real clinical/utilization data with external cost
benchmarks when claims-level cost data isn't available.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports", "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

RAW_PATH = "/mnt/user-data/uploads/diabetic_data.csv"


# ==============================================================================
# 1. LOAD & CLEAN
# ==============================================================================
def load_and_clean(path=RAW_PATH):
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)

    # Keep one encounter per patient (first encounter) to avoid
    # within-patient correlation inflating the sample
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")

    return df


# ==============================================================================
# 2. CMS-STYLE "PLANNED / NON-INDEX" ADMISSION FILTER
#    (proxy, adapted from CMS Planned Readmission Algorithm logic:
#    exclude admissions unlikely to represent an unplanned acute event)
# ==============================================================================
def apply_planned_admission_filter(df):
    # Exclude: newborns (admission_type 4), expired patients (discharge
    # dispositions 11,19,20,21 = various forms of death), and hospice
    # (13,14) -- none of these are eligible for a "readmission" outcome
    # in the way CMS defines it.
    exclude_admission_types = [4]  # Newborn
    exclude_discharge = [11, 13, 14, 19, 20, 21]  # expired / hospice

    mask = (~df["admission_type_id"].isin(exclude_admission_types)) & (
        ~df["discharge_disposition_id"].isin(exclude_discharge)
    )
    return df[mask].copy()


# ==============================================================================
# 3. CLAIMS-STYLE ACUITY PROXY ENGINEERING
#    (built from real utilization/clinical fields, analogous in spirit to
#    the CPT/J-code/revenue-code proxies used with true claims data)
# ==============================================================================
def engineer_acuity_proxies(df):
    df = df.copy()

    df["high_prior_utilization"] = (df["number_inpatient"] >= 1).astype(int)
    df["recent_ed_use"] = (df["number_emergency"] >= 1).astype(int)
    df["high_diagnosis_burden"] = (df["number_diagnoses"] >= 9).astype(int)
    df["polypharmacy"] = (df["num_medications"] >= 15).astype(int)
    df["long_stay"] = (df["time_in_hospital"] >= 5).astype(int)
    df["insulin_change"] = (df["insulin"].isin(["Up", "Down"])).astype(int)
    df["any_med_change"] = (df["change"] == "Ch").astype(int)
    df["poor_glycemic_control"] = (
        df["A1Cresult"].isin([">7", ">8"])
    ).astype(int)
    df["high_lab_intensity"] = (df["num_lab_procedures"] >= 45).astype(int)

    df["claims_acuity_score"] = (
        df["high_prior_utilization"] * 2.0
        + df["recent_ed_use"] * 1.8
        + df["high_diagnosis_burden"] * 1.3
        + df["polypharmacy"] * 1.0
        + df["long_stay"] * 1.2
        + df["insulin_change"] * 0.8
        + df["poor_glycemic_control"] * 0.7
        + df["high_lab_intensity"] * 0.5
    )

    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)

    return df


# ==============================================================================
# 4. MODEL: PREDICT 30-DAY READMISSION FROM ACUITY PROXIES
# ==============================================================================
FEATURES = [
    "high_prior_utilization", "recent_ed_use", "high_diagnosis_burden",
    "polypharmacy", "long_stay", "insulin_change", "any_med_change",
    "poor_glycemic_control", "high_lab_intensity", "number_diagnoses",
    "num_medications", "time_in_hospital",
]


def fit_model(df):
    X = df[FEATURES].fillna(0)
    y = df["readmitted_30"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)

    coefs = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0]),
    }).sort_values("odds_ratio", ascending=False)

    return model, scaler, X_test, y_test.values, probs, auc, ap, coefs


# ==============================================================================
# 5. DECISION-THRESHOLD SENSITIVITY / ECONOMIC NET-BENEFIT ANALYSIS
#    Cost assumptions are ILLUSTRATIVE (see module docstring), based on
#    published ranges for diabetes-related 30-day readmission costs and
#    typical care-transition outreach program costs.
# ==============================================================================
COST_FALSE_POSITIVE = 75      # cost of an unnecessary outreach/care-transition call
COST_FALSE_NEGATIVE = 11000   # avg. cost of an unprevented 30-day diabetes readmission (illustrative, literature-informed)
BENEFIT_TRUE_POSITIVE = 6500  # net benefit of successfully preventing a readmission via intervention (illustrative)


def threshold_sensitivity(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        net_savings = (tp * BENEFIT_TRUE_POSITIVE) - (fp * COST_FALSE_POSITIVE) - (fn * 0)
        # Note: FN "cost" is a missed-opportunity cost, not a program spend;
        # shown separately for transparency rather than folded into net_savings
        missed_opportunity_cost = fn * COST_FALSE_NEGATIVE

        rows.append({
            "threshold": round(th, 2),
            "sensitivity": round(sensitivity, 3),
            "ppv": round(ppv, 3),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "program_net_savings_usd": round(net_savings, 2),
            "missed_opportunity_cost_usd": round(missed_opportunity_cost, 2),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 6. EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("Loading and cleaning real UCI diabetic encounters dataset...")
    df = load_and_clean()
    print(f"  {len(df):,} unique-patient encounters after dedup")

    df_filtered = apply_planned_admission_filter(df)
    print(f"  {len(df_filtered):,} encounters after excluding newborn/expired/hospice")

    df_proc = engineer_acuity_proxies(df_filtered)
    df_proc.to_csv(os.path.join(DATA_DIR, "processed_acuity_cohort.csv"), index=False)

    print(f"  30-day readmission rate in cohort: {df_proc['readmitted_30'].mean():.1%}")

    model, scaler, X_test, y_test, probs, auc, ap, coefs = fit_model(df_proc)
    print(f"\nModel performance (held-out test set):")
    print(f"  ROC-AUC: {auc:.3f}")
    print(f"  PR-AUC (average precision): {ap:.3f}")

    coefs.to_csv(os.path.join(SCRIPT_DIR, "reports", "acuity_odds_ratios.csv"), index=False)
    print("\nTop acuity proxy odds ratios:")
    print(coefs.to_string(index=False))

    sens_df = threshold_sensitivity(y_test, probs)
    sens_df.to_csv(os.path.join(SCRIPT_DIR, "reports", "threshold_sensitivity.csv"), index=False)
    print("\nThreshold sensitivity (excerpt):")
    print(sens_df[["threshold", "sensitivity", "ppv", "program_net_savings_usd"]].to_string(index=False))

    # --- Figure 1: ROC + PR curves ---
    fpr, tpr, _ = roc_curve(y_test, probs)
    prec, rec, _ = precision_recall_curve(y_test, probs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color="#d62728", linewidth=2, label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Model Discrimination: 30-Day Readmission Risk (Acuity Proxy Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_pr_curves.png"), dpi=150)
    plt.close()

    # --- Figure 2: Threshold sensitivity / net savings curve ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(sens_df["threshold"], sens_df["program_net_savings_usd"] / 1000,
              marker="o", color="#2ca02c", linewidth=2)
    plt.axhline(0, color="gray", linewidth=1, linestyle="--")
    plt.title("Estimated Program Net Savings Across Decision Thresholds")
    plt.xlabel("Probability Decision Threshold")
    plt.ylabel("Net Savings ($ Thousands, illustrative cost assumptions)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "threshold_sensitivity_curve.png"), dpi=150)
    plt.close()

    # --- Figure 3: Acuity proxy odds ratios ---
    plt.figure(figsize=(8, 5))
    plot_df = coefs.sort_values("odds_ratio")
    plt.barh(plot_df["feature"], plot_df["odds_ratio"], color="#9467bd")
    plt.axvline(1.0, color="gray", linewidth=1, linestyle="--")
    plt.title("Adjusted Odds Ratios: Acuity Proxies vs. 30-Day Readmission")
    plt.xlabel("Odds Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "acuity_odds_ratios.png"), dpi=150)
    plt.close()

    print(f"\nDone. Figures saved to {REPORTS_DIR}")
