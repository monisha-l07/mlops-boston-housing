import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
DATA_PATH = "data/HousingData.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")


# Separate features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Define models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# MLflow experiment
mlflow.set_experiment("Boston Housing Regression")


# Store results
results = []

best_model = None
best_model_name = None
best_rmse = float("inf")


# Train each model
for model_name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training: {model_name}")
    print("=" * 50)

    # Handle missing values
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])

    with mlflow.start_run(run_name=model_name):

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        predictions = pipeline.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        r2 = r2_score(y_test, predictions)

        # Display metrics
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

        # Log parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"]
        )

        # Store results
        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        # Select best model using RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_model_name = model_name


# Display comparison
results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# Save best model
best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

joblib.dump(best_model, best_model_path)


print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model : {best_model_name}")
print(f"RMSE  : {best_rmse:.4f}")
print(f"Saved : {best_model_path}")
