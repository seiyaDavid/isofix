"""
Stock Anomaly Detection API

This module provides a FastAPI-based REST API for detecting anomalies in stock price data.
It uses Isolation Forest algorithm for anomaly detection and MLflow for model management.

Endpoints:
    - /detect_anomalies/: Detect anomalies using existing or new models
    - /force_retrain/: Force retrain models for all stocks

The API expects CSV files with a 'Date' column and one or more stock price columns.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import uvicorn
import os
from src.training.trainer import ModelTrainer
from src.utils.mlflow_utils import MLFlowManager
from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger
from typing import Dict, List, Annotated
import json
from datetime import datetime

logger = setup_logger("api")
app = FastAPI(title="Stock Anomaly Detection API")

# Initialize MLflow and trainer
os.makedirs("mlruns", exist_ok=True)
mlflow_manager = MLFlowManager("config/config.yaml")
trainer = ModelTrainer("config/config.yaml", "config/hyperparameters.yaml")


def validate_input_data(data: pd.DataFrame) -> None:
    """
    Validate the input data format and content.

    Args:
        data (pd.DataFrame): Input DataFrame containing stock data

    Raises:
        HTTPException: If data validation fails with appropriate error message
            - 400: Empty data provided
            - 400: Missing Date column
            - 400: No stock columns found
    """
    # Check for empty data
    if data.empty:
        raise HTTPException(status_code=400, detail="Empty data provided")

    # Check for Date column
    if "Date" not in data.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'Date' column")

    # Check for stock columns
    stocks = [col for col in data.columns if col != "Date"]
    if not stocks:
        raise HTTPException(status_code=400, detail="No stock columns found")


@app.post("/detect_anomalies/")
async def detect_anomalies(
    file: Annotated[UploadFile, File(description="CSV file containing stock data")],
) -> Dict:
    """
    Detect anomalies in stock data using existing models or training new ones.

    Args:
        file: CSV file with columns: 'Date' and one or more stock columns
            Format: Date,STOCK1,STOCK2,...
            Example: 2023-01-01,100.5,45.6,...

    Returns:
        Dict containing:
            - date: Most recent date in the data
            - anomalous_stocks: List of anomalous stocks and their scores
            - all_anomalies: List of all anomalies detected in the data

    Raises:
        HTTPException:
            - 400: Invalid data format or content
            - 500: Processing or model error
    """
    try:
        # Read CSV file
        try:
            content = await file.read()
            data = pd.read_csv(pd.io.common.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error reading CSV file: {str(e)}"
            )

        # Validate input data - move this before the try-except block
        if data.empty:
            raise HTTPException(status_code=400, detail="Empty data provided")

        if "Date" not in data.columns:
            raise HTTPException(
                status_code=400, detail="CSV must contain a 'Date' column"
            )

        stocks = [col for col in data.columns if col != "Date"]
        if not stocks:
            raise HTTPException(status_code=400, detail="No stock columns found")

        # Get most recent date
        most_recent_date = data["Date"].iloc[-1]

        # Process each stock
        all_anomalies = []
        anomalous_stocks = {}

        for stock in stocks:
            try:
                model = mlflow_manager.get_model(stock)

                if model is None:
                    logger.info(f"Training new model for {stock}")
                    anomalies, predictions, scores = trainer.train_stock_model(
                        stock, data
                    )
                    if not anomalies.empty:
                        # Convert Pandas objects to native Python types
                        anomalies_dict = {
                            "Date": pd.to_datetime(anomalies["Date"])
                            .dt.strftime("%Y-%m-%d")
                            .tolist(),
                            f"{stock}_Value": anomalies[f"{stock}_Value"].tolist(),
                            f"{stock}_PctChange": anomalies[
                                f"{stock}_PctChange"
                            ].tolist(),
                            f"{stock}_AnomalyScore": anomalies[
                                f"{stock}_AnomalyScore"
                            ].tolist(),
                            f"{stock}_IsAnomaly": anomalies[
                                f"{stock}_IsAnomaly"
                            ].tolist(),
                        }
                        all_anomalies.append(anomalies_dict)

                        if predictions[-1] == -1:
                            anomalous_stocks[stock] = {
                                "is_anomaly": True,
                                "value": float(data[stock].iloc[-1]),
                                "anomaly_score": float(scores[-1]),
                            }
                else:
                    logger.info(f"Using existing model for {stock}")
                    training_data, _, csv_values, csv_dates = DataLoader(
                        None
                    ).prepare_stock_data(data, stock)
                    predictions = model.predict(training_data)
                    scores = model.score_samples(training_data)

                    if any(predictions == -1):
                        if predictions[-1] == -1:
                            anomalous_stocks[stock] = {
                                "is_anomaly": True,
                                "value": float(csv_values.iloc[-1]),
                                "anomaly_score": float(scores[-1]),
                            }

                        # Convert Pandas objects to native Python types
                        mask = predictions == -1
                        anomalies_dict = {
                            "Date": pd.to_datetime(data.loc[mask, "Date"])
                            .dt.strftime("%Y-%m-%d")
                            .tolist(),
                            f"{stock}_Value": csv_values[mask].tolist(),
                            f"{stock}_PctChange": training_data.iloc[mask, 0].tolist(),
                            f"{stock}_AnomalyScore": scores[mask].tolist(),
                            f"{stock}_IsAnomaly": [True] * sum(mask),
                        }
                        all_anomalies.append(anomalies_dict)

            except Exception as e:
                logger.error(f"Error processing {stock}: {str(e)}")
                continue

        return {
            "date": pd.to_datetime(data["Date"].iloc[-1]).strftime("%Y-%m-%d"),
            "anomalous_stocks": anomalous_stocks,
            "all_anomalies": all_anomalies,
        }

    except HTTPException as he:
        # Re-raise HTTP exceptions with their original status code
        raise he
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/force_retrain/")
async def force_retrain(
    file: Annotated[UploadFile, File(description="CSV file containing stock data")],
) -> Dict:
    """
    Force retrain models for all stocks and return only anomalous stocks for the most recent date
    """
    try:
        # Read CSV file
        try:
            content = await file.read()
            data = pd.read_csv(pd.io.common.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error reading CSV file: {str(e)}"
            )

        validate_input_data(data)

        # Get most recent date
        most_recent_date = data["Date"].iloc[-1]

        # Process each stock
        anomalous_results = {}
        for stock in data.columns:
            try:
                # Force retrain model
                logger.info(f"Force retraining model for {stock}")
                anomalies, predictions, scores = trainer.train_stock_model(stock, data)

                # Only include if most recent data point is anomalous
                if predictions[-1] == -1:
                    anomalous_results[stock] = {
                        "anomaly_score": float(scores[-1]),
                        "value": float(data[stock].iloc[-1]),
                    }

            except Exception as e:
                logger.error(f"Error processing {stock}: {str(e)}")
                continue

        return (
            {"date": most_recent_date, "anomalous_stocks": anomalous_results}
            if anomalous_results
            else {"date": most_recent_date, "anomalous_stocks": {}}
        )

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Stock Anomaly Detection API",
        "endpoints": {
            "detect_anomalies": "/detect_anomalies/",
            "force_retrain": "/force_retrain/",
        },
        "documentation": {"swagger": "/docs", "redoc": "/redoc"},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
