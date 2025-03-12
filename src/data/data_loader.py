import pandas as pd
from typing import Tuple
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")

"""
Data Loading and Preprocessing Module

This module handles all data-related operations including:
    - Loading stock data from CSV files
    - Data validation and cleaning
    - Feature preparation for model training
    - Data formatting for inference
"""


class DataLoader:
    """
    Handles data loading and preprocessing for stock anomaly detection.

    The class provides methods for:
        - Loading stock data
        - Preparing features for model training
        - Validating data format and content
        - Converting data for model inference
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Load the stock data from csv file"""
        try:
            logger.info(f"Loading data from {self.file_path}")
            df = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def prepare_stock_data(
        self, df: pd.DataFrame, stock: str
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for a specific stock.

        Args:
            df (pd.DataFrame): Raw CSV data with Date and stock columns
            stock (str): Stock column name to process

        Returns:Data
            Tuple containing:
                - DataFrame with absolute percentage changes (for training)
                - Series of dates for percentage changes
                - Series of original stock values from CSV
                - Series of dates from CSV
        """
        try:
            logger.info(f"Preparing data for stock: {stock}")

            # 1. Get original values from CSV
            df["Date"] = pd.to_datetime(df["Date"])
            csv_values = df[stock].copy()  # Keep original CSV values
            csv_dates = df["Date"]

            # 2. Calculate percentage changes
            pct_change = df[stock].pct_change()

            # 3. Make percentage changes absolute (positive)
            abs_pct_change = pct_change.abs()

            # 4. Create DataFrame for training (dropping first row with NaN)
            training_data = pd.DataFrame({stock: abs_pct_change})

            # 5. Align indices for training data and original data
            training_data = training_data.reset_index(drop=True)
            training_dates = pd.Series(range(len(training_data)))

            logger.info(f"Successfully prepared data for {stock}")
            return training_data, training_dates, csv_values, csv_dates

        except Exception as e:
            logger.error(f"Error preparing data for {stock}: {str(e)}")
            raise
