import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("house price data.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ==============================
# Remove Unnecessary Columns
# ==============================

drop_columns = ["date", "street"]

for col in drop_columns:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# ==============================
# Features & Target
# ==============================

X = df.drop("price", axis=1)
y = df["price"]

# ==============================
# Numerical & Categorical Columns
# ==============================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X.select_dtypes(include=["object"]).columns

print("\nNumerical Features")
print(numeric_features)

print("\nCategorical Features")
print(categorical_features)

# ==============================
# Preprocessing
# ==============================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==============================
# Model
# ==============================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==============================
# Train Model
# ==============================

print("\nTraining Model...\n")

pipeline.fit(X_train, y_train)

print("Training Completed!")

# ==============================
# Predictions
# ==============================

predictions = pipeline.predict(X_test)

# ==============================
# Evaluation Metrics
# ==============================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mse ** 0.5

r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:,.2f}")

print(f"MSE  : {mse:,.2f}")

print(f"RMSE : {rmse:,.2f}")

print(f"R²   : {r2:.4f}")

# ==============================
# Save Model
# ==============================

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModel Saved Successfully!")

# ==============================
# Save Metrics
# ==============================

metrics = {
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2
}

with open("metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("Metrics Saved Successfully!")