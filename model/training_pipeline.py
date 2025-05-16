# model/training_pipeline.py
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def load_data(path='data/insurance.csv'):
    return pd.read_csv(path)

def preprocess_and_train(df):
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    X = df.drop(['PremiumPrice', 'Height', 'Weight'], axis=1)
    y = df['PremiumPrice']

    numeric = ['Age', 'BMI']
    categorical = X.select_dtypes(include=['bool', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'model/model.pkl')
    print("✅ Model and preprocessor saved.")

if __name__ == '__main__':
    df = load_data()
    preprocess_and_train(df)
