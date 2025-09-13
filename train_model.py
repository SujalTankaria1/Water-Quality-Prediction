# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Step 2: Load Dataset
data = pd.read_csv("dataset/water_potability.csv")
print("Shape:", data.shape)


# Step 3: Data Cleaning
imputer = SimpleImputer(strategy='median')
data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

plt.figure(figsize=(10,6))
sns.heatmap(data_cleaned.corr(), annot=True, cmap="coolwarm")
plt.show()

sns.countplot(x="Potability", data=data_cleaned)
plt.show()

# Step 4: Preprocessing
X = data_cleaned.drop("Potability", axis=1)
y = data_cleaned["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Define Base Models
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Step 6: Hyperparameter Tuning with GridSearch
param_grid_rf = {
    "n_estimators": [200, 300, 400],
    "max_depth": [15, 20, None],
    "min_samples_split": [2, 5, 10]
}

param_grid_gb = {
    "n_estimators": [200, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5]
}

print("🔎 Tuning RandomForest...")
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
print("✅ Best RF Params:", grid_rf.best_params_)

print("🔎 Tuning GradientBoosting...")
grid_gb = GridSearchCV(gb, param_grid_gb, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid_gb.fit(X_train_scaled, y_train)
best_gb = grid_gb.best_estimator_
print("✅ Best GB Params:", grid_gb.best_params_)

# Step 7: Voting Classifier (Best Models)
ensemble = VotingClassifier(
    estimators=[("rf", best_rf), ("gb", best_gb)],
    voting="soft"
)

ensemble.fit(X_train_scaled, y_train)

# Step 8: Evaluation
y_pred = ensemble.predict(X_test_scaled)

print("🎯 Final Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Save Model & Scaler
os.makedirs("models", exist_ok=True)
joblib.dump(ensemble, "models/water_quality_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("🎉 Optimized Model & Scaler Saved Successfully!")
# Step 10: Feature Importance (from RandomForest & GradientBoosting)
importances_rf = best_rf.feature_importances_
importances_gb = best_gb.feature_importances_

# Average importance (since we’re using ensemble)
avg_importances = (importances_rf + importances_gb) / 2

# Create DataFrame for plotting
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": avg_importances
}).sort_values(by="Importance", ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("🔍 Feature Importance in Water Quality Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

print("📊 Top Important Features:\n", importance_df.head())
