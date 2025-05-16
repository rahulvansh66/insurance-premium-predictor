# app/api.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from utils import calculate_bmi 

app = Flask(__name__)
model = joblib.load("model/model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df['BMI'] = calculate_bmi(df['Weight'] , df['Height'])  #df['Weight'] / ((df['Height'] / 100) ** 2)
    df = df.drop(['Height', 'Weight'], axis=1)
    
    prediction = model.predict(df)[0]
    return jsonify({'predicted_premium': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
