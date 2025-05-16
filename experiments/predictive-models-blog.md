---

From Data to Insights: Predicting Insurance Premiums with Machine Learning
Predicting insurance premiums is a complex and nuanced task. It involves understanding customer profiles, risk factors, and historical data to make accurate predictions. In this blog, I'll walk you through a machine learning-driven approach to predict insurance premiums, highlighting the importance of preprocessing, model selection, and interpretability. This blog is based on an experiment I conducted using Python and key machine learning libraries like Scikit-learn and XGBoost.

---

The Problem: Predicting Insurance Premiums
Insurance companies rely on data-driven models to calculate premiums for their customers. These premiums depend on various factors, such as age, BMI, medical history, and family health background. Our goal is to build a machine learning pipeline to predict insurance premiums (PremiumPrice) using these features.

---

Step 1: Data Preprocessing
The first step in any machine learning journey is preparing the data. While the dataset used in this experiment did not have missing values, I ensured it was ready for modeling by applying the following steps:
Feature Engineering: I calculated the Body Mass Index (BMI) using height and weight, as BMI is often a predictor of health risks.
Encoding Categorical Variables: Categorical features like the number of major surgeries were label-encoded.
Scaling Numerical Features: Features like age, height, weight, and BMI were standardized to ensure models could perform optimally.

Here's a snippet of the preprocessing function:
def preprocess_data(df):
  df['Height_m'] = df['Height'] / 100
  df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
  df.drop(columns=['Height_m'], inplace=True)
  bool_cols = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants',
  'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily']
  df[bool_cols] = df[bool_cols].astype(int)
  df['NumberOfMajorSurgeries'] = LabelEncoder().fit_transform(df['NumberOfMajorSurgeries'])
  scaler = StandardScaler()
  scale_cols = ['Age', 'Height', 'Weight', 'BMI']
  df[scale_cols] = scaler.fit_transform(df[scale_cols])
  X = df.drop(columns=['PremiumPrice'])
  y = df['PremiumPrice']
  return X, y

---

Step 2: Model Selection and Evaluation
To predict PremiumPrice, I experimented with several machine learning models:
Linear Regression: A baseline model for capturing linear relationships.
Decision Tree Regressor: A non-linear model for better flexibility.
Random Forest Regressor: An ensemble model that combines multiple decision trees.
Gradient Boosting Regressor: A boosting algorithm that iteratively minimizes errors.

Each model was evaluated using cross-validation, and performance metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score were calculated.
def evaluate_model(model, X, y, name="Model"):
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores, mae_scores, r2_scores = [], [], []
for train_index, val_index in kf.split(X):
X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
model.fit(X_train_fold, y_train_fold)
y_pred = model.predict(X_val_fold)
rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
r2_scores.append(r2_score(y_val_fold, y_pred))
print(f"\n📊 {name} Evaluation:")
print(f"Average RMSE: {np.mean(rmse_scores):.2f}")
print(f"Average MAE: {np.mean(mae_scores):.2f}")
print(f"Average R²: {np.mean(r2_scores):.4f}")

---

Step 3: Hyperparameter Tuning
While Random Forest and Gradient Boosting performed well, I wanted to fine-tune their performance using Randomized Search. This approach explored a range of hyperparameters, such as the number of estimators, maximum depth, and learning rate, to find the best configuration.
param_grid = {
'n_estimators': [100, 200, 300],
'max_depth': [3, 5, 7],
'learning_rate': [0.01, 0.1, 0.2],
'subsample': [0.8, 1],
'colsample_bytree': [0.8, 1]
}
search = RandomizedSearchCV(
xgb.XGBRegressor(random_state=42),
param_distributions=param_grid,
n_iter=20,
scoring='r2',
cv=5,
n_jobs=-1
)
search.fit(X_train, y_train)
print("Best Parameters:", search.best_params_)

---

Step 4: Repeated Experiments for Robustness
To ensure the models were robust and not overfitting to a particular train-test split, I repeated the experiments with different random seeds. This allowed me to evaluate the consistency of the model's performance.

---

Step 5: Final Model Selection
After evaluating the models, I used two approaches to select the final model:
Most Frequently Occurring Hyperparameters: This approach prioritizes consistent performance across multiple runs.
Best R² on Test Set: This approach selects the model with the highest test R² score.

from collections import Counter
def get_best_final_model(algorithm_name, model_class, param_list):
param_counts = Counter([frozenset(p.items()) for p in param_list])
best_params_frozen = param_counts.most_common(1)[0][0]
best_params = dict(best_params_frozen)
return best_params

---

Step 6: Prediction Intervals
To quantify uncertainty in predictions, I calculated prediction intervals using a statistical approach. These intervals provided a range within which the actual premium price was likely to fall, giving confidence in the predictions.
def prediction_interval(model, X_test, y_test, confidence=0.95):
  predictions = model.predict(X_test)
  residuals = predictions - y_test
  std_error = np.std(residuals)
  z_score = norm.ppf((1 + confidence) / 2)
  margin = z_score * std_error
  lower = predictions - margin
  upper = predictions + margin
  return predictions, lower, upper

---

Step 7: Interpretability and Explainability
Once the final model was trained, I focused on making it interpretable:
Feature Importance: I visualized feature importance to understand which features had the most impact on predictions.
Permutation Importance: This technique helped me validate the importance of features by measuring the drop in performance when a feature's values were shuffled.
SHAP Values: SHAP (SHapley Additive exPlanations) values provided insights into how individual features influenced predictions.

import shap
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

---

Key Takeaways
Preprocessing Matters: Feature engineering, scaling, and encoding are critical for improving model performance.
Model Selection is Iterative: Experiment with multiple models and fine-tune them to find the best fit.
Robustness Over Perfection: Consistency across multiple experiments is more valuable than a single high-performing model.
Explainability is Key: Use feature importance, permutation importance, and SHAP values to make your models transparent and trustworthy.

---

Conclusion
Building a machine learning model to predict insurance premiums is a journey that requires careful planning, experimentation, and validation. By following the steps outlined in this blog, you can create robust models that not only perform well but also provide valuable insights into the factors driving predictions.
Have you worked on a similar problem? I'd love to hear your thoughts and experiences in the comments below!