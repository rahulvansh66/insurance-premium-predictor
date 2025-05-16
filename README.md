# Insurance Premium Predictor

This project aims to analyze and predict insurance premium costs based on demographic and health-related factors. Using exploratory data analysis (EDA), hypothesis testing, and machine learning models, we uncover key insights and build a predictive pipeline for insurance premiums.

## End-to-End project

- **Exploratory Data Analysis (EDA):** Analyze distributions, relationships, and correlations between features and premium costs.
- **Hypothesis Testing:** Validate the impact of health conditions on premium pricing.
- **Machine Learning Pipeline:** Predict insurance premiums using models like Random Forest and Gradient Boosting. 
- **Explainability and Interpretation of the Model:** It's quite important to understand which features are highly affecting to the premium price. 
- **Web Application:** A Flask-based API serves predictions, while a Streamlit-based web app provides an interactive user interface.

To understand more you can check out my blogs:
[Analysis and Hypothesis Testing of Insurance Premium Costs](https://medium.com/@rahulvansh66/analyzing-insurance-premium-costs-6a9fedbe8b5c)
[Predicting Insurance Premiums with Machine Learning](https://medium.com/@rahulvansh66/predicting-insurance-premiums-with-machine-learning-cf40234b26f0)

## Dataset Overview

The dataset contains 11 features, including demographic factors (e.g., age, height, weight) and health-related attributes (e.g., diabetes, chronic diseases). The target variable, `PremiumPrice`, represents the insurance premium cost.

## Web Application Architecture

1. **Flask API:**
   - The Flask app ([`app/api.py`](app/api.py)) loads a pre-trained machine learning model (`model.pkl`) and exposes a `/predict` endpoint.
   - The endpoint accepts JSON input, processes the data (e.g., calculates BMI), and returns the predicted premium.

2. **Streamlit Web App:**
   - The Streamlit app ([`app/streamlit_app.py`](app/streamlit_app.py)) provides a user-friendly interface for inputting data.
   - It sends the input data as a POST request to the Flask API and displays the predicted premium on success.

### How It Works:
- The Streamlit app interacts with the Flask API to fetch predictions and display them to the user.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rahulvansh66/insurance-premium-predictor.git
   cd insurance-premium-predictor

