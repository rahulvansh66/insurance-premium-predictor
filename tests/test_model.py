import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load('model/model.pkl')
    df = pd.DataFrame([{
        "Age": 35, "Diabetes": False, "BloodPressureProblems": False,
        "AnyTransplants": False, "AnyChronicDiseases": False,
        "Height": 170, "Weight": 70, "KnownAllergies": False,
        "HistoryOfCancerInFamily": False, "NumberOfMajorSurgeries": 0,
        "BMI": 24.22
    }])
    prediction = model.predict(df)
    assert len(prediction) == 1
