import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("logistic_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = pd.DataFrame([{
        "gender": request.form.get("gender", "Male"),
        "age": float(request.form["age"]),
        "hypertension": int(request.form["hypertension"]),
        "heart_disease": int(request.form["heart_disease"]),
        "ever_married": request.form.get("ever_married", "No"),
        "work_type": request.form.get("work_type", "Private"),
        "Residence_type": request.form.get("Residence_type", "Urban"),
        "avg_glucose_level": float(request.form["avg_glucose_level"]),
        "bmi": float(request.form["bmi"]),
        "smoking_status": request.form.get("smoking_status", "never smoked")
    }])

    proba = model.predict_proba(input_data)[0][1]

    result = (
        "Nguy cơ đột quỵ CAO 🚨"
        if proba >= 0.5
        else "Nguy cơ đột quỵ THẤP ✅"
    )

    return render_template(
        "index.html",
        prediction_text=f"{result} (Độ tin cậy: {proba:.2%})"
    )

if __name__ == "__main__":
    app.run(debug=True)
