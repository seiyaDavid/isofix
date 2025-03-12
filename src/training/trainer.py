import mlflow
import pandas as pd
import yaml
from src.models.isolation_forest import StockAnomalyDetector
from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger
import numpy as np

logger = setup_logger("trainer")

"""
Stock Anomaly Detection Model Trainer

This module handles the training of anomaly detection models for stock data.
It implements:
    - Isolation Forest algorithm
    - Hyperparameter optimization using Optuna
    - Model persistence with MLflow
    - Training workflow management
"""


class ModelTrainer:
    """
    Handles the training of anomaly detection models for stock data.

    Attributes:
        config (dict): General configuration parameters
        hyperparameters (dict): Model hyperparameter ranges
        mlflow_manager (MLFlowManager): MLflow interface for model management

    The trainer:
        - Optimizes hyperparameters for each stock
        - Trains Isolation Forest models
        - Saves models and metadata to MLflow
        - Handles the complete training workflow
    """

    def __init__(self, config_path: str, hp_config_path: str):
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
            self.hp_config_path = hp_config_path
            logger.info("ModelTrainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ModelTrainer: {str(e)}")
            raise

    def train_stock_model(self, stock: str, data: pd.DataFrame):
        """Train model for a specific stock"""
        try:
            logger.info(f"Starting training process for {stock}")
            data_loader = DataLoader(None)

            # Get training data and original values
            training_data, training_dates, csv_values, csv_dates = (
                data_loader.prepare_stock_data(data, stock)
            )

            detector = StockAnomalyDetector(stock, self.hp_config_path)

            with mlflow.start_run(run_name=f"{stock}_model") as run:
                mlflow.set_tag("stock", stock)

                # Train on percentage changes
                model, best_params = detector.train(training_data)
                predictions = model.predict(training_data)
                anomaly_scores = model.score_samples(training_data)

                # Log model with proper name
                mlflow.sklearn.log_model(
                    model,
                    f"models/{stock}_model",
                    registered_model_name=f"{stock}_model",
                )

                # Get indices where anomalies were detected
                anomaly_indices = np.where(predictions == -1)[0]

                # Create anomalies DataFrame using original CSV values
                anomalies = pd.DataFrame(
                    {
                        "Date": csv_dates[anomaly_indices],
                        f"{stock}_Value": csv_values[anomaly_indices],
                        f"{stock}_PctChange": training_data.iloc[
                            anomaly_indices, 0
                        ].values,
                        f"{stock}_AnomalyScore": anomaly_scores[anomaly_indices],
                        f"{stock}_IsAnomaly": True,
                    }
                )

                logger.info(f"Found {len(anomalies)} anomalies for {stock}")
                return anomalies, predictions, anomaly_scores

        except Exception as e:
            logger.error(f"Error in train_stock_model for {stock}: {str(e)}")
            raise
