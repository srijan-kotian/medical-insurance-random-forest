import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load Data
df = pd.read_csv('insurance.csv')

# 2. One-Hot Encoding (The "Anti-Bias" Move)
# We use drop_first=True to avoid the dummy variable trap
df = pd.get_dummies(df, columns=['sex', 'region'], drop_first=True)
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# 3. Features & Target
X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model with Statistical Validation
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Metrics for Researchers
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"--- RESEARCH METRICS ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"5-Fold Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 6. Save the model and the feature list (Required for the App)
joblib.dump(model, 'medical_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')
print("Model and Feature Schema saved.")