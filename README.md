## Screenshots / Evidence

### 1. MLflow Experiments

MLflow was used to track and compare the machine learning experiments.

<img width="922" height="761" alt="mlflow-experiments" src="https://github.com/user-attachments/assets/8860f328-00eb-4f45-bfb5-9f8344466aa4" />



### 2. Registered Model

The best-performing Random Forest model was registered in MLflow.

<img width="931" height="752" alt="mlflow-registered-model png" src="https://github.com/user-attachments/assets/58808f99-c5cd-47b2-838e-11b0a10782ae" />



### 3. DVC Tracking

DVC was used for dataset and pipeline tracking. The training pipeline was successfully reproduced using `dvc repro`.

<img width="786" height="841" alt="dvc-tracking" src="https://github.com/user-attachments/assets/b853891f-2ec1-4590-a5b3-3f7e272ea185" />



### 4. GitHub Actions CI

GitHub Actions successfully executed the CI pipeline after pushing the project to GitHub.

<img width="650" height="441" alt="github-actions png" src="https://github.com/user-attachments/assets/fb9966f9-d8ce-4b61-93d9-80f858cedc55" />



### 5. FastAPI Prediction Endpoint

The FastAPI `/predict` endpoint successfully returned a house-price prediction.

<img width="896" height="672" alt="fastapi-prediction" src="https://github.com/user-attachments/assets/333f94b2-b561-42c8-89f6-82d4a5dd9f5a" />
