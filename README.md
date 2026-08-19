# Real-World Readmission Risk Analytics: Acuity Proxies & Economic Threshold Sensitivity

A claims-style readmission risk framework built on **real, publicly available inpatient
encounter data** (UCI "Diabetes 130-US hospitals for years 1999–2008" dataset,
101,766 encounters). Rather than relying on true billing/claims fields (which this
dataset doesn't contain), the project engineers **acuity proxies from real clinical
and utilization variables** — modeled after how claims-based acuity flags (CPT,
revenue codes, J-codes) are typically used in payer/health-system analytics.

## Project Overview

1. **CMS-style planned/non-index admission filtering**: excludes newborn admissions
   and in-hospital deaths/hospice discharges from the readmission-eligible cohort,
   consistent with the logic behind CMS's Planned Readmission Algorithm.
2. **Acuity proxy engineering**: derives severity/instability flags from real fields
   — prior inpatient utilization, recent ED use, diagnosis burden, polypharmacy,
   length of stay, insulin dose changes, glycemic control, and lab intensity.
3. **Logistic regression risk model**: predicts 30-day readmission from acuity
   proxies; reports adjusted odds ratios, ROC-AUC, and PR-AUC on a held-out test set.
4. **Decision-threshold sensitivity / economic net-benefit analysis**: evaluates
   how different probability cutoffs for a hypothetical care-transition outreach
   program would trade off sensitivity, PPV, and estimated net savings.

## Important Note on Cost Assumptions

This dataset contains **no billing or claims cost fields**. The dollar figures used
in the threshold sensitivity analysis (cost of an outreach call, cost of a missed
readmission, benefit of a prevented readmission) are **illustrative assumptions**,
informed by published ranges in readmission-cost literature — not observed data.
This is disclosed explicitly in the code and results, and mirrors a common real-world
constraint: health economics teams frequently have to pair real clinical/utilization
data with external cost benchmarks when claims-level cost data isn't available.

## Key Results

- Cohort: 69,964 encounters after excluding newborns, in-hospital deaths, and hospice
  discharges (from 71,518 unique-patient encounters).
- 30-day readmission rate in cohort: **8.97%**
- Model discrimination: **ROC-AUC 0.586**, PR-AUC (average precision) 0.120 —
  modest, and honestly reported. This reflects a real, well-documented limitation
  of claims/EHR-derived acuity proxies alone: they capture *some* readmission
  signal but leave most variance unexplained, which is consistent with published
  literature on this exact dataset and task.
- Strongest acuity signal: **prior inpatient utilization** (OR 1.23) and
  **diagnosis burden** (OR 1.18) — patients with recent hospitalizations and more
  comorbidities were meaningfully more likely to be readmitted within 30 days.
- The net-savings curve shows diminishing returns at higher decision thresholds:
  because the model's sensitivity drops off sharply above a ~0.20 probability
  cutoff, program-level "net savings" (under the illustrative cost assumptions)
  concentrate almost entirely in the lowest-threshold, highest-sensitivity range.

## Figures

### Model Discrimination: ROC & Precision-Recall Curves
![ROC and Precision-Recall Curves](reports/figures/roc_pr_curves.png)

The model performs modestly better than chance (ROC-AUC 0.586). The
precision-recall curve reflects the class imbalance in the data (~9% base
readmission rate) — precision stays low across most of the recall range,
consistent with the limited discriminative power of acuity proxies alone.

### Adjusted Odds Ratios: Acuity Proxies vs. 30-Day Readmission
![Acuity Odds Ratios](reports/figures/acuity_odds_ratios.png)

Prior inpatient utilization and diagnosis burden show the strongest positive
association with 30-day readmission. Notably, polypharmacy and poor glycemic
control were not independently predictive once other acuity factors were
controlled for.

### Estimated Program Net Savings Across Decision Thresholds
![Threshold Sensitivity Curve](reports/figures/threshold_sensitivity_curve.png)

Under the illustrative cost assumptions, estimated net savings from a
hypothetical outreach program are concentrated almost entirely at low
probability thresholds and collapse to near-zero above ~0.20, driven by the
model's sharp drop in sensitivity at higher cutoffs.

## Repository Structure
```
readmission_project/
├── data/
│   └── processed_acuity_cohort.csv       # Cleaned, filtered, feature-engineered cohort
├── reports/
│   ├── acuity_odds_ratios.csv            # Full logistic regression output
│   ├── threshold_sensitivity.csv         # Full threshold sensitivity table
│   └── figures/
│       ├── roc_pr_curves.png
│       ├── threshold_sensitivity_curve.png
│       └── acuity_odds_ratios.png
├── readmission_acuity_analysis.py         # Full analysis pipeline
└── README.md
```

## Tools
Python, pandas, numpy, scikit-learn, matplotlib

## Limitations
- This is an observational, retrospective analysis; associations do not establish
  causation.
- Acuity proxies are derived from encounter-level clinical/administrative fields,
  not true billing claims data (no CPT/J-code/revenue-code fields exist in this
  dataset).
- Cost/benefit figures are illustrative, not sourced from the dataset itself, and
  are clearly labeled as such throughout.
- Data spans 1999–2008; inpatient diabetes management and readmission-reduction
  practices have evolved substantially since.
- This analysis is for educational and portfolio purposes and should not be used
  to guide real clinical or program-funding decisions.

**Author:** Shruthi Nagaraju, MD, MHA (DHA Candidate)

