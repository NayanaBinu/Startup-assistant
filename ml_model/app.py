import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

# Load the saved model
model = joblib.load("startup_success_predictor_v3_optimized.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        funding_total_usd = float(request.form["funding_total_usd"])
        funding_rounds = int(request.form["funding_rounds"])
        age_in_days = int(request.form["age_in_days"])
        funding_duration_days = int(request.form["funding_duration_days"])
        funding_velocity = float(request.form["funding_velocity"])
        country_code = request.form["country_code"]
        primary_category = request.form["primary_category"]

        # Create DataFrame for model input
        input_data = pd.DataFrame([{
            "funding_total_usd": funding_total_usd,
            "funding_rounds": funding_rounds,
            "age_in_days": age_in_days,
            "funding_duration_days": funding_duration_days,
            "funding_velocity": funding_velocity,
            "country_code": country_code,
            "primary_category": primary_category
        }])

        # Predict
        prediction_proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        result = "Successful Startup 🚀" if prediction == 1 else "Not Successful ❌"
        return render_template("index.html",
                               prediction_text=f"Prediction: {result} (Success Probability: {prediction_proba:.2f})")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
