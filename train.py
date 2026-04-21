# ==============================
# Salary Prediction Model Training
# ==============================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("Salary_Dataset_DataScienceLovers.csv")

# ==============================
# 2. Data Cleaning
# ==============================

df = df.dropna()
df = df.drop_duplicates()

# ==============================
# 3. Feature Selection
# ==============================

X = df[['Rating', 'Salaries Reported']]
y = df['Salary']

# ==============================
# 4. Feature Scaling
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 5. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# 6. Define Models
# ==============================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVM": SVR(),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# ==============================
# 7. Train & Evaluate
# ==============================

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {"R2": r2, "MAE": mae}

# ==============================
# 8. Show Results
# ==============================

print("\n📊 Model Performance:")
for model_name, metrics in results.items():
    print(f"{model_name} → R2: {metrics['R2']:.4f}, MAE: {metrics['MAE']:.2f}")

# ==============================
# 9. Select Best Model
# ==============================

best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")

# ==============================
# 10. Save Model & Scaler
# ==============================

pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\n✅ Model and scaler saved successfully!")

# ==============================
# 11. Test Prediction
# ==============================

sample = [[4.5, 10]]  # Rating, Salaries Reported
sample_scaled = scaler.transform(sample)

prediction = best_model.predict(sample_scaled)
print(f"\n🔮 Sample Prediction: {prediction[0]:.2f}")
