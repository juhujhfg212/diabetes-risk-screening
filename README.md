\# Diabetes Risk Screening App



A machine learning web application that estimates diabetes risk from health indicators. Final project for CpE Elective 1 — Applied Machine Learning and Software Development.



\- \*\*Author:\*\* Frederick Galinato

\- \*\*Course / Section:\*\* BSCpE CSCU3

\- \*\*School Year:\*\* 2026



\---



\## 1. Problem Statement



Diabetes is a leading cause of disability and death in the Philippines, and many cases go undiagnosed until complications appear because routine screening is not always accessible. This project provides a low-cost, offline screening tool that estimates a person's risk from 21 self-reported health indicators and directs higher-risk users to consult a healthcare professional.



This tool is not a medical diagnosis. It is an educational screening prototype.



\## 2. Dataset



\- \*\*Source:\*\* CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, distributed via the UCI Machine Learning Repository (dataset ID 891).

\- \*\*URL:\*\* https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

\- \*\*Records:\*\* 253,680 original observations.

\- \*\*Features:\*\* 21 health, lifestyle, and demographic indicators.

\- \*\*Target:\*\* Diabetes\_binary — 0 = no diabetes, 1 = diabetes or prediabetes.

\- \*\*Class distribution:\*\* 218,334 (86.1%) non-diabetic, 35,346 (13.9%) diabetic. Handled with class\_weight set to balanced.

\- \*\*Preprocessing:\*\* No missing values. Features already numeric-coded. StandardScaler applied for the logistic regression baseline only; tree-based models used raw features.



\## 3. Machine Learning Approach



Three models were trained and compared on the same 80/20 stratified split (202,944 training / 50,736 test records).



| Model | Accuracy | Recall (diabetic) | F1 (diabetic) |

|---|---|---|---|

| Logistic Regression (baseline) | 73.15% | 76.11% | 0.4413 |

| Random Forest | 82.44% | 43.63% | 0.4091 |

| Hist Gradient Boosting | 71.86% | 79.71% | 0.4411 |



Model selection rationale: For a screening tool, missing a diabetic case (false negative) is more costly than flagging a healthy person for follow-up (false positive). We therefore selected the model with the highest recall on the diabetic class — Hist Gradient Boosting at 79.71% — even though Random Forest achieved higher raw accuracy. This tradeoff is deliberate and documented.



\## 4. Application Features



The Flask web app supports all required features plus several additional ones.



Required:



\- Input: Multi-step form covering all 21 model features, with human-readable labels, dropdowns for categorical fields, bounded number inputs, and a short explanation under each field.

\- Prediction: Each submission is passed through the trained Hist Gradient Boosting model.

\- Results: A three-band risk categorization (Lower / Moderate / Higher) with the estimated probability and tailored advice.

\- Validation: Empty fields and non-numeric input produce clear error messages. Numerical fields are range-bounded in HTML.

\- Usable interface: Multi-step flow (intro to 4 form steps to result) with progress bar, back navigation, and a prominent "not a medical diagnosis" disclaimer on both the intro and result pages.



Additional:



\- Answer review: The result page shows every submitted answer in plain English so users can verify or edit.

\- PDF export: A "Print / Save as PDF" button generates a clean, printable record the user can share with a doctor. Print-specific CSS hides interactive buttons from the PDF.

\- Session persistence: Answers are stored per-user across steps; the Back button preserves previous inputs.



\## 5. Tech Stack



\- Python 3.11

\- pandas — data loading and preprocessing

\- scikit-learn — model training, evaluation, splitting, scaling

\- Flask — web server and Jinja2 template rendering

\- joblib — model serialization



\## 6. How to Run Locally



Install dependencies:



&#x20;   pip install -r requirements.txt



Download the dataset from https://archive.ics.uci.edu/static/public/891/data.csv and save it as cdc\_diabetes.csv in this folder.



Train the model (creates model.joblib, features.joblib, scaler.joblib):



&#x20;   python train\_model.py



Start the app:



&#x20;   python app.py



Open http://127.0.0.1:5000 in your browser.



\## 7. Project Structure



&#x20;   diabetes\_project/

&#x20;     app.py                    Flask server (multi-step flow)

&#x20;     train\_model.py            Model training, comparison, and export

&#x20;     check\_duplicates.py       Diagnostic script (see Section 9)

&#x20;     templates/

&#x20;       intro.html              Landing page with warning

&#x20;       form.html               Multi-step form page

&#x20;       result.html             Result page with answer review + PDF export

&#x20;     model.joblib              Trained Hist Gradient Boosting model

&#x20;     features.joblib           Ordered feature list

&#x20;     scaler.joblib             StandardScaler (for baseline model)

&#x20;     requirements.txt          Python dependencies

&#x20;     README.md                 This file

&#x20;     .gitignore                Excludes dataset and cache



\## 8. Limitations



This tool is a screening starting point, not a diagnostic tool.



\- Self-reported inputs. BMI is what the user types, not a measurement. All other features are survey responses. No blood tests or clinical measurements are used.

\- US-based training data. The model was trained on BRFSS 2015 (US respondents). It has not been validated on Philippine populations.

\- Precision on flagged cases is roughly 30 percent. Of 100 people the tool flags as higher risk, about 30 actually have diabetes or prediabetes. This is acceptable for screening but means "Higher risk" is a prompt to get checked, not a prediction.

\- Age dominates the prediction. In the CDC data, diabetes prevalence in the 18-24 group is very low. Young users with multiple risk factors may receive lower scores than their clinical profile would suggest. This is a property of the training distribution, not a coding bug.

\- Not a substitute for clinical judgment. Every result page carries a visible disclaimer stating this is not a medical diagnosis.



\## 9. Notes on Earlier Iteration



An early version of this project attempted a phishing URL detection task using the PhiUSIIL dataset. Diagnostic analysis (check\_duplicates.py) revealed that the URL-only feature subset collapsed 235,795 records into only about 35,491 unique feature patterns, causing train/test leakage and inflated metrics (F1 above 0.99). The project pivoted to the CDC diabetes dataset, which has no such leakage. The diagnostic script is included in the repository for transparency.



\## 10. AI Tools Disclosure



The following AI tools were used during development:



\- Claude (Anthropic) — used for code scaffolding, explanation of ML concepts, debugging assistance, and UI templating. All code was reviewed, executed, and validated by the author. The author is able to explain every part of the code and the model selection rationale.



\## 11. Acknowledgments



\- Dataset: CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, via the UCI Machine Learning Repository.

\- Course: CpE Elective 1 — Applied Machine Learning and Software Development.

