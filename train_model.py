# train_model.py  (CDC Diabetes Health Indicators)
# Trains and compares two ML models. Saves the better one for the Flask app.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# ---- 1. Load ----
print("Loading dataset...")
df = pd.read_csv("cdc_diabetes.csv")
print(f"Raw rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# ---- 2. Define features and target ----
# Drop ID (not a feature) and keep the rest as inputs.
FEATURES = [c for c in df.columns if c not in ("ID", "Diabetes_binary")]
TARGET = "Diabetes_binary"

print(f"\nUsing {len(FEATURES)} features: {FEATURES}")
print(f"Label distribution:\n{df[TARGET].value_counts()}")

# ---- 3. Split ----
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

# ---- 4. Scale (fit on train only) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 5. Logistic Regression (baseline) ----
print("\nTraining Logistic Regression (baseline)...")
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)
log_pred = log_reg.predict(X_test_scaled)

# ---- 6. Random Forest (comparison) ----
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ---- 6b. Hist Gradient Boosting (third comparison) ----
print("Training Hist Gradient Boosting...")
hgb = HistGradientBoostingClassifier(
    max_iter=300, random_state=42, class_weight="balanced"
)
hgb.fit(X_train, y_train)  # HGB handles unscaled features fine
hgb_pred = hgb.predict(X_test)


# ---- 7. Evaluate ----
def report(name, y_true, y_pred):
    print(f"\n{name}:")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1       : {f1_score(y_true, y_pred):.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred, target_names=["No Diabetes", "Diabetes/Prediabetes"]))
    return f1_score(y_true, y_pred)

f1_log = report("Logistic Regression", y_test, log_pred)
f1_rf = report("Random Forest", y_test, rf_pred)
f1_hgb = report("Hist Gradient Boosting", y_test, hgb_pred)

# Compare all three on F1 and RECALL.
# For a screening tool, recall matters more than F1: we'd rather
# flag a healthy person for a follow-up test than miss a diabetic case.
results = {
    "Logistic Regression":   (log_reg, scaler, f1_log, recall_score(y_test, log_pred)),
    "Random Forest":         (rf,      None,   f1_rf,  recall_score(y_test, rf_pred)),
    "Hist Gradient Boosting":(hgb,     None,   f1_hgb, recall_score(y_test, hgb_pred)),
}

# Selection rule: highest recall wins. Ties broken by F1.
best_name = max(results, key=lambda k: (results[k][3], results[k][2]))
best_f1, best_recall = results[best_name][2], results[best_name][3]
best_model, best_scaler = results[best_name][0], results[best_name][1]

print(f"\nWinner on recall (screening priority): {best_name}")
print(f"  F1 = {best_f1:.4f}   Recall = {best_recall:.4f}")
joblib.dump(best_model, "model.joblib")
joblib.dump(FEATURES, "features.joblib")
if best_scaler is not None:
    joblib.dump(best_scaler, "scaler.joblib")
    print("Saved: model.joblib, features.joblib, scaler.joblib")
else:
    print("Saved: model.joblib, features.joblib (no scaler needed for RF)")