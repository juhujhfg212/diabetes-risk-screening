\# Diabetes Risk Screening App



A machine learning web application that estimates diabetes risk from health indicators. Final project for CpE Elective 1 — Applied Machine Learning and Software Development.



\*\*Author:\*\* \[Frederick Galinato ]

\*\*Course / Section:\*\* \[BSCpE CSCU3]

\*\*School Year:\*\* 2026



\---



\## 1. Problem Statement



Diabetes is a leading cause of disability and death in the Philippines, and many cases go undiagnosed until complications appear because routine screening is not always accessible. This project provides a \*\*low-cost, offline screening tool\*\* that estimates a person's risk from 21 self-reported health indicators and directs higher-risk users to consult a healthcare professional.



\*\*This tool is not a medical diagnosis.\*\* It is an educational screening prototype.



\## 2. Dataset



\- \*\*Source:\*\* CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, distributed via the UCI Machine Learning Repository (dataset ID 891).

\- \*\*URL:\*\* https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

\- \*\*Records:\*\* 253,680 original observations.

\- \*\*Features:\*\* 21 health, lifestyle, and demographic indicators.

\- \*\*Target:\*\* `Diabetes\_binary` — 0 = no diabetes, 1 = diabetes or prediabetes.

\- \*\*Class distribution:\*\* 218,334 (86.1%) non-diabetic, 35,346 (13.9%) diabetic. Handled with `class\_weight="balanced"`.

\- \*\*Preprocessing:\*\* No missing values. Features already numeric-coded. `StandardScaler` applied for the logistic regression baseline only; tree-based models used raw features.



\## 3. Machine Learning Approach



Three models were trained and compared on the same 80/20 stratified split (202,944 training / 50,736 test records).



| Model | Accuracy | Recall (diabetic) | F1 (diabetic) |

|---|---|---|---|

| Logistic Regression (baseline) | 73.15% | 76.11% | 0.4413 |

| Random Forest | 82.44% | 43.63% | 0.4091 |

| \*\*Hist Gradient Boosting\*\* | 71.86% | \*\*79.71%\*\* | 0.4411 |



\*\*Model selection rationale:\*\* For a \*screening\* tool, missing a diabetic case (false negative) is more costly than flagging a healthy person for follow-up (false positive). We therefore selected the model with the highest \*\*recall on the diabetic class\*\* — Hist Gradient Boosting at 79.71% — even though Random Forest achieved higher raw accuracy. This tradeoff is deliberate and documented.



\## 4. Application Features



The Flask web app supports all required features plus several additional ones:



\*\*Required:\*\*

\- \*\*Input:\*\* Multi-step form covering all 21 model features, with human-readable labels, dropdowns for categorical fields, bounded number inputs, and a short explanation under each field.

\- \*\*Prediction:\*\* Each submission is passed through the trained Hist Gradient Boosting model.

\- \*\*Results:\*\* A three-band risk categorization (Lower / Moderate / Higher) with the estimated probability and tailored advice.

\- \*\*Validation:\*\* Empty fields and non-numeric input produce clear error messages. Numerical fields are range-bounded in HTML.

\- \*\*Usable interface:\*\* Multi-step flow (intro → 4 form steps → result) with progress bar, back navigation, and a prominent "not a medical diagnosis" disclaimer on both the intro and result pages.



\*\*Additional:\*\*

\- \*\*Answer review:\*\* The result page shows every submitted answer in plain English so users can verify or edit.

\- \*\*PDF export:\*\* A "Print / Save as PDF" button generates a clean, printable record the user can share with a doctor. Print-specific CSS hides interactive buttons from the PDF.

\- \*\*Session persistence:\*\* Answers are stored per-user across steps; the Back button preserves previous inputs.



\## 5. Tech Stack



\- \*\*Python 3.11\*\*

\- \*\*pandas\*\* — data loading and preprocessing

\- \*\*scikit-learn\*\* — model training, evaluation, splitting, scaling

\- \*\*Flask\*\* — web server and Jinja2 template rendering

\- \*\*joblib\*\* — model serialization



\## 6. How to Run Locally



```bash

\# 1. Install dependencies

pip install -r requirements.txt



\# 2. Download the dataset

\# Save as cdc\_diabetes.csv in this folder.

\# URL: https://archive.ics.uci.edu/static/public/891/data.csv



\# 3. Train the model (creates model.joblib, features.joblib, scaler.joblib)

python train\_model.py



\# 4. Start the app

python app.py



\# 5. Open in browser

\# http://127.0.0.1:5000

