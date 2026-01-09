# import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    '''
    Render prediction result on HTML
    '''

    # Lấy dữ liệu từ form theo đúng thứ tự model đã train
    age = float(request.form['age'])
    hypertension = float(request.form['hypertension'])
    heart_disease = float(request.form['heart_disease'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])

    final_features = pd.DataFrame([{
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi
    }])

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    # Vì stroke là 0/1 nên diễn giải kết quả
    if output >= 0.5:
        result = "Nguy cơ đột quỵ cao"
    else:
        result = "Nguy cơ đột quỵ thấp"

    return render_template(
        'index.html',
        prediction_text=f'Dự đoán: {result} (value = {output})'
    )

if __name__ == "__main__":
    app.run(debug=True)
