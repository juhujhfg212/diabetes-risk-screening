# app.py — Flask server for the Diabetes Risk Screening app
# Multi-step flow: intro -> 4 form steps -> result.

from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-string-before-submission"

# ---- Load model + features ----
model = joblib.load("model.joblib")
FEATURES = joblib.load("features.joblib")
print(f"Loaded model. Expecting {len(FEATURES)} features.")

# ---- Feature specs ----
FIELD_SPECS = {
    # --- Step 1: About You ---
    "Sex":      {"label": "Sex", "type": "select",
                 "options": [(0, "Female"), (1, "Male")],
                 "hint": "Biological sex", "group": 1},
    "Age":      {"label": "Age category", "type": "select",
                 "options": [(1,"18-24"),(2,"25-29"),(3,"30-34"),(4,"35-39"),
                             (5,"40-44"),(6,"45-49"),(7,"50-54"),(8,"55-59"),
                             (9,"60-64"),(10,"65-69"),(11,"70-74"),(12,"75-79"),(13,"80+")],
                 "hint": "Your age bracket", "group": 1},
    "Education":{"label": "Education level", "type": "select",
                 "options": [(1,"Never attended / kindergarten"),(2,"Elementary"),
                             (3,"Some high school"),(4,"High school graduate"),
                             (5,"Some college / technical"),(6,"College graduate")],
                 "hint": "Highest level completed", "group": 1},
    "Income":   {"label": "Household income level", "type": "select",
                 "options": [(1,"< $10k"),(2,"$10k-15k"),(3,"$15k-20k"),(4,"$20k-25k"),
                             (5,"$25k-35k"),(6,"$35k-50k"),(7,"$50k-75k"),(8,"$75k+")],
                 "hint": "Approximate yearly household income", "group": 1},

    # --- Step 2: Health Conditions ---
    "HighBP":               {"label": "High blood pressure", "type": "yesno",
                             "hint": "Have you ever been told you have high blood pressure?", "group": 2},
    "HighChol":             {"label": "High cholesterol", "type": "yesno",
                             "hint": "Have you ever been told you have high cholesterol?", "group": 2},
    "CholCheck":            {"label": "Cholesterol check (last 5 years)", "type": "yesno",
                             "hint": "Have you had a cholesterol check in the past 5 years?", "group": 2},
    "HeartDiseaseorAttack": {"label": "Heart disease or heart attack", "type": "yesno",
                             "hint": "Have you ever had heart disease or a heart attack?", "group": 2},
    "Stroke":               {"label": "Stroke", "type": "yesno",
                             "hint": "Have you ever had a stroke?", "group": 2},
    "DiffWalk":             {"label": "Difficulty walking / climbing stairs", "type": "yesno",
                             "hint": "Do you have serious difficulty walking or climbing stairs?", "group": 2},
    "BMI":                  {"label": "BMI (Body Mass Index)", "type": "number",
                             "min": 10, "max": 70, "step": 0.1,
                             "hint": "Weight in kg divided by height in m squared. Normal: 18.5–24.9",
                             "group": 2},

    # --- Step 3: Lifestyle ---
    "Smoker":            {"label": "Smoker", "type": "yesno",
                          "hint": "Have you smoked at least 100 cigarettes in your life?", "group": 3},
    "PhysActivity":      {"label": "Physical activity (past 30 days)", "type": "yesno",
                          "hint": "Any physical activity or exercise in the past 30 days?", "group": 3},
    "Fruits":            {"label": "Eats fruit daily", "type": "yesno",
                          "hint": "Do you eat fruit one or more times per day?", "group": 3},
    "Veggies":           {"label": "Eats vegetables daily", "type": "yesno",
                          "hint": "Do you eat vegetables one or more times per day?", "group": 3},
    "HvyAlcoholConsump":{"label": "Heavy alcohol consumption", "type": "yesno",
                          "hint": "Adult men: >14 drinks/week. Adult women: >7 drinks/week.",
                          "group": 3},

    # --- Step 4: Healthcare & Well-being ---
    "AnyHealthcare":  {"label": "Has healthcare coverage", "type": "yesno",
                       "hint": "Do you have any health insurance or coverage?", "group": 4},
    "NoDocbcCost":    {"label": "Skipped doctor due to cost", "type": "yesno",
                       "hint": "In the past year, did cost stop you from seeing a doctor?", "group": 4},
    "GenHlth":        {"label": "General health", "type": "select",
                       "options": [(1,"Excellent"),(2,"Very good"),(3,"Good"),(4,"Fair"),(5,"Poor")],
                       "hint": "Overall, how would you rate your health?", "group": 4},
    "MentHlth":       {"label": "Poor mental health days (past 30)", "type": "number",
                       "min": 0, "max": 30, "step": 1,
                       "hint": "Number of days you felt mentally unwell", "group": 4},
    "PhysHlth":       {"label": "Poor physical health days (past 30)", "type": "number",
                       "min": 0, "max": 30, "step": 1,
                       "hint": "Number of days you felt physically unwell", "group": 4},
}

# Sanity check
missing = [f for f in FEATURES if f not in FIELD_SPECS]
if missing:
    raise RuntimeError(f"Missing FIELD_SPECS entries for: {missing}")

STEP_TITLES = {
    1: "About You",
    2: "Health Conditions",
    3: "Lifestyle",
    4: "Healthcare & Well-being",
}
STEP_SUBTITLES = {
    1: "Basic demographic information. These help the model calibrate your baseline risk.",
    2: "Any existing conditions or measurements. If you don't know your BMI, estimate it — the model tolerates approximation.",
    3: "Daily habits that strongly influence diabetes risk.",
    4: "Healthcare access and how you've been feeling recently.",
}

# ---- Helpers ----
def features_for_group(group_id):
    """Return the list of feature names belonging to a given step."""
    return [f for f in FEATURES if FIELD_SPECS[f]["group"] == group_id]

def format_answer(feature, value):
    """Turn a raw numeric answer into a human-readable string."""
    spec = FIELD_SPECS[feature]
    if spec["type"] == "yesno":
        return "Yes" if value == 1 else "No"
    if spec["type"] == "select":
        for val, text in spec["options"]:
            if val == value:
                return text
        return str(value)
    # number field
    if float(value).is_integer():
        return str(int(value))
    return str(value)

# ---- Routes ----
@app.route("/")
def intro():
    return render_template("intro.html", total_steps=len(STEP_TITLES))

@app.route("/form/<int:step>", methods=["GET", "POST"])
def form(step):
    if step < 1 or step > len(STEP_TITLES):
        return redirect(url_for("form", step=1))

    step_features = features_for_group(step)
    error = None
    values = session.get("answers", {})

    if request.method == "POST":
        try:
            step_values = {}
            for f in step_features:
                raw = request.form.get(f, "").strip()
                if raw == "":
                    raise ValueError(f"'{FIELD_SPECS[f]['label']}' is required.")
                try:
                    step_values[f] = float(raw)
                except ValueError:
                    raise ValueError(f"'{FIELD_SPECS[f]['label']}' must be a number.")
            answers = session.get("answers", {})
            answers.update(step_values)
            session["answers"] = answers
            if step < len(STEP_TITLES):
                return redirect(url_for("form", step=step + 1))
            else:
                return redirect(url_for("result"))
        except ValueError as e:
            error = str(e)
            values = {k: v for k, v in request.form.items()}

    return render_template(
        "form.html",
        step=step,
        total_steps=len(STEP_TITLES),
        step_title=STEP_TITLES[step],
        step_subtitle=STEP_SUBTITLES[step],
        step_features=step_features,
        field_specs=FIELD_SPECS,
        error=error,
        values=values,
    )

@app.route("/result")
def result():
    answers = session.get("answers", {})
    if not all(f in answers for f in FEATURES):
        return redirect(url_for("form", step=1))

    # Build the row in the EXACT order the model expects
    row = [answers[f] for f in FEATURES]
    X = np.array([row])
    prob = model.predict_proba(X)[0, 1]
    prob_pct = round(float(prob) * 100, 1)

    if prob_pct >= 60:
        band, css = "Higher risk", "result-warn"
        advice = ("This result suggests elevated risk indicators. Please consult "
                  "a healthcare professional for a proper screening.")
    elif prob_pct >= 30:
        band, css = "Moderate risk", "result-moderate"
        advice = ("Some risk indicators are present. Consider a check-up with a "
                  "healthcare professional, especially if you have other risk factors.")
    else:
        band, css = "Lower risk", "result-good"
        advice = ("This result suggests lower risk based on the indicators provided. "
                  "Keep up healthy habits and continue regular check-ups.")

    # Build review list for the answer table
    answer_list = [
        {"label": FIELD_SPECS[f]["label"], "display": format_answer(f, answers[f])}
        for f in FEATURES
    ]

    return render_template(
        "result.html",
        probability=prob_pct,
        band=band,
        css=css,
        advice=advice,
        answer_list=answer_list,
    )

@app.route("/restart")
def restart():
    session.clear()
    return redirect(url_for("intro"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)