"""
Isolation Forest Model for Stock Anomaly Detection

This module implements the Isolation Forest algorithm for detecting anomalies in stock data.
Features:
    - Automated hyperparameter optimization using Optuna
    - Configurable parameter ranges
    - Model training and prediction
    - Anomaly score calculation
"""

import optuna
import mlflow
import numpy as np
from sklearn.ensemble import IsolationForest
import yaml
from src.utils.logger import setup_logger
from typing import Tuple, Dict, Any

logger = setup_logger("isolation_forest")


class StockAnomalyDetector:
    """
    Implements Isolation Forest algorithm with hyperparameter optimization.

    Attributes:
        stock_name (str): Name of the stock being analyzed
        hp_config (dict): Hyperparameter configuration ranges
        best_params (dict): Best hyperparameters found during optimization
        model (IsolationForest): Trained Isolation Forest model
    """

    def __init__(self, stock_name: str, config_path: str):
        """
        Initialize the anomaly detector.

        Args:
            stock_name (str): Name of the stock to analyze
            config_path (str): Path to hyperparameter configuration YAML
        """
        self.stock_name = stock_name
        try:
            with open(config_path, "r") as file:
                self.hp_config = yaml.safe_load(file)["isolation_forest"]
            logger.info(f"Initialized StockAnomalyDetector for {stock_name}")
            self.X_train = None
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optimization objective for Optuna.

        Args:
            trial (optuna.Trial): Optuna trial object

        Returns:
            float: Optimization metric (anomaly detection score)
        """
        try:
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    self.hp_config["n_estimators"]["low"],
                    self.hp_config["n_estimators"]["high"],
                ),
                "max_samples": trial.suggest_float(
                    "max_samples",
                    self.hp_config["max_samples"]["low"],
                    self.hp_config["max_samples"]["high"],
                ),
                "contamination": trial.suggest_float(
                    "contamination",
                    self.hp_config["contamination"]["low"],
                    self.hp_config["contamination"]["high"],
                ),
                "max_features": trial.suggest_float(
                    "max_features",
                    self.hp_config["max_features"]["low"],
                    self.hp_config["max_features"]["high"],
                ),
                "bootstrap": trial.suggest_categorical(
                    "bootstrap", self.hp_config["bootstrap"]["options"]
                ),
            }

            model = IsolationForest(**params, random_state=42)
            model.fit(self.X_train)

            scores = model.score_samples(self.X_train)
            return -np.mean(np.abs(scores))
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            raise

    def train(self, X: np.ndarray) -> Tuple[IsolationForest, Dict[str, Any]]:
        """
        Train the Isolation Forest model with optimized hyperparameters.

        Args:
            X (np.ndarray): Training data

        Returns:
            Tuple containing:
                - Trained IsolationForest model
                - Dictionary of best hyperparameters
        """
        try:
            logger.info(f"Starting training for {self.stock_name}")
            self.X_train = X

            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=20)

            best_params = study.best_params
            logger.info(f"Best parameters found: {best_params}")

            final_model = IsolationForest(**best_params, random_state=42)
            final_model.fit(X)

            logger.info(f"Successfully trained model for {self.stock_name}")
            return final_model, best_params
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
