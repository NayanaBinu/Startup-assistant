from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("startup_xgb_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # ===== USER INPUTS =====
    age = float(request.form["age_startup_year"])
    funding = float(request.form["funding_total_usd"])
    rounds = float(request.form["funding_rounds"])
    milestones = float(request.form["milestones"])
    participants = float(request.form["avg_participants"])
    relationships = float(request.form["relationships"])

    # ===== Tier Engineering =====
    if relationships <= 5:
        tier = 4
    elif relationships <= 10:
        tier = 3
    elif relationships <= 16:
        tier = 2
    else:
        tier = 1

    # ===== State Encoding =====

    # Indian startups → always otherstate
    is_CA = 0
    is_NY = 0
    is_TX = 0
    is_MA = 0
    is_otherstate = 1

    # ===== Category Encoding =====
    category = request.form["category"]

    is_software = 1 if category == "software" else 0
    is_web = 1 if category == "web" else 0
    is_mobile = 1 if category == "mobile" else 0
    is_enterprise = 1 if category == "enterprise" else 0
    is_advertising = 1 if category == "advertising" else 0
    is_gamesvideo = 1 if category == "gamesvideo" else 0
    is_ecommerce = 1 if category == "ecommerce" else 0
    is_biotech = 1 if category == "biotech" else 0
    is_consulting = 1 if category == "consulting" else 0
    is_othercategory = 1 if category == "other" else 0

    # ===== Investment Encoding =====
    has_VC = int(request.form["has_VC"])
    has_angel = int(request.form["has_angel"])
    has_Seed = int(request.form["has_Seed"])

    # ===== FULL FEATURE DICTIONARY =====
    features_dict = {

        'age_first_funding_year': 0,
        'age_last_funding_year': 0,
        'age_first_milestone_year': 0,
        'age_last_milestone_year': 0,
        'funding_rounds': rounds,
        'funding_total_usd': funding,
        'milestones': milestones,

        'is_CA': is_CA,
        'is_NY': is_NY,
        'is_MA': is_MA,
        'is_TX': is_TX,
        'is_otherstate': is_otherstate,

        'is_software': is_software,
        'is_web': is_web,
        'is_mobile': is_mobile,
        'is_enterprise': is_enterprise,
        'is_advertising': is_advertising,
        'is_gamesvideo': is_gamesvideo,
        'is_ecommerce': is_ecommerce,
        'is_biotech': is_biotech,
        'is_consulting': is_consulting,
        'is_othercategory': is_othercategory,

        'has_VC': has_VC,
        'has_angel': has_angel,
        'has_roundA': 0,
        'has_roundB': 0,
        'has_roundC': 0,
        'has_roundD': 0,

        'avg_participants': participants,
        'is_top500': 0,
        'has_RoundABCD': 0,
        'has_Investor': 0,
        'has_Seed': has_Seed,
        'invalid_startup': 0,

        'age_startup_year': age,
        'tier_relationships': tier
    }

    # Convert to DataFrame
    features_df = pd.DataFrame([features_dict])

    # Prediction
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1]

    if prediction == 1:
        result = "🚀 Startup Likely to SUCCEED"
    else:
        result = "⚠️ Startup Likely to FAIL"

    return render_template(
        "result.html",
        result=result,
        probability=round(probability * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)