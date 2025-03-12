import mlflow
import yaml
from src.utils.logger import setup_logger
import os
from typing import Optional, Any

logger = setup_logger("mlflow_utils")

"""
MLflow Model Management Utilities

This module handles all MLflow-related operations including:
    - Model versioning
    - Model storage and retrieval
    - Experiment tracking
    - Model metadata management
    - Run history tracking

The module ensures consistent model management across training and inference.
"""


class MLFlowManager:
    """
    Manages MLflow operations for model versioning and tracking.

    This class provides a centralized interface for all MLflow operations,
    ensuring consistent model management across the application.

    Attributes:
        config (dict): Configuration parameters for MLflow including:
            - tracking_uri: Location for storing MLflow data
            - experiment_name: Name of the MLflow experiment

    Methods:
        get_model: Retrieve latest model for a specific stock
        log_model: Save a trained model with metadata
        get_run_history: Retrieve training history for a stock
    """

    def __init__(self, config_path: str):
        """
        Initialize MLflow manager with configuration.

        Args:
            config_path (str): Path to YAML configuration file containing:
                mlflow:
                    tracking_uri: Path to MLflow tracking directory
                    experiment_name: Name for the experiment
        """
        try:
            logger.info(f"Initializing MLFlowManager with config: {config_path}")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)

            # Set up MLflow tracking
            tracking_uri = self.config["mlflow"]["tracking_uri"]
            self.experiment_name = self.config["mlflow"]["experiment_name"]

            # Ensure mlruns directory exists
            os.makedirs(tracking_uri, exist_ok=True)

            # Set tracking URI and get/create experiment
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment_name)

            # Get experiment ID
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id

            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
            logger.info(f"MLflow experiment name: {self.experiment_name}")
            logger.info(f"MLflow experiment ID: {self.experiment_id}")
            logger.info("MLflow manager initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing MLflow manager: {str(e)}")
            raise

    def get_model(self, stock: str) -> Optional[Any]:
        """
        Retrieve the latest trained model for a specific stock.

        Args:
            stock (str): Stock symbol/name to retrieve model for

        Returns:
            Optional[Any]: Latest trained model if exists, None otherwise

        Note:
            Returns the most recently trained model based on run timestamp
        """
        try:
            # Search for runs with this stock tag
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.stock = '{stock}'",
                order_by=["start_time DESC"],
            )

            if not runs.empty:
                latest_run = runs.iloc[0]
                model = mlflow.sklearn.load_model(
                    f"runs:/{latest_run.run_id}/models/{stock}_model"
                )
                logger.info(f"Successfully loaded model for {stock}")
                return model

            logger.info(f"No existing model found for {stock}")
            return None

        except Exception as e:
            logger.warning(f"Error loading model for {stock}: {str(e)}")
            return None
