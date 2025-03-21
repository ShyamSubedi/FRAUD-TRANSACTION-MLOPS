# Fraud Detection API - MLOps Demo

This project demonstrates an end-to-end MLOps pipeline using a machine learning model trained to detect fraudulent financial transactions.

## About the Project

- Model: XGBoost
- API Framework: FastAPI
- Dataset: Synthetic financial dataset (Kaggle)
- Purpose: Demonstrate model deployment, API creation, and logging

## Disclaimer

This is a test model trained on synthetic data for demonstration purposes only. It is not intended for use in real-world fraud detection systems.

## Project Files

- `fraud_api_v3.py`: FastAPI script that serves the model
- `fraud_detection_v3_model.json`: Trained XGBoost model
- `logs.db`: SQLite database storing prediction logs
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## How to Run the API

1. Install the required packages:

   pip install -r requirements.txt

2. Start the FastAPI server:

   uvicorn fraud_api_v3:app --reload

3. Open your browser at:

   http://127.0.0.1:8000

## Example Prediction Request

POST `/predict/` with JSON:

{
  "amount": 1000000.0
}

Example response:

{
  "fraud_prediction": 1,
  "fraud_probability": 0.9998
}

## API Endpoints

- `/health`: Check if API is running
- `/predict/`: Send a transaction amount and get fraud prediction
- `/logs/`: View stored prediction logs

## MLOps Tasks Demonstrated

- Training and saving a model
- Deploying model with FastAPI
- Logging predictions to a database
- Making predictions through a REST API
"""
