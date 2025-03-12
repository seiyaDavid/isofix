"""
Stock Anomaly Detection Streamlit Application

This module provides a web interface for the stock anomaly detection system.
It allows users to:
    - Upload stock price data
    - Train/retrain anomaly detection models
    - Visualize anomalies in interactive plots
    - Download detected anomalies as CSV

The application uses:
    - Streamlit for the web interface
    - MLflow for model management
    - Plotly for interactive visualizations
"""

import streamlit as st
import pandas as pd
import mlflow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from src.training.trainer import ModelTrainer
from src.utils.mlflow_utils import MLFlowManager
from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger
import os
from typing import List, Dict

logger = setup_logger("streamlit_app")


def create_stock_plot(data, stock, predictions, anomaly_scores):
    """Create plotly figure for a stock"""
    fig = go.Figure()

    # Add original values
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data[stock],
            mode="lines",
            name="Original Values",
            line=dict(color="blue"),
        )
    )

    # Add anomalies
    anomaly_indices = np.where(predictions == -1)[0]
    fig.add_trace(
        go.Scatter(
            x=data["Date"].iloc[anomaly_indices],
            y=data[stock].iloc[anomaly_indices],
            mode="markers",
            name="Anomalies",
            marker=dict(color="red", size=10),
        )
    )

    fig.update_layout(
        title=f"{stock} Values and Anomalies",
        xaxis_title="Date",
        yaxis_title="Value",
        showlegend=True,
    )

    return fig


def main():
    """
    Main application function that handles:
        - File upload interface
        - Model training controls
        - Progress tracking
        - Results visualization
        - Anomaly data export

    The function provides two modes:
        1. Force Retrain: Retrain all models regardless of existing ones
        2. Selective Retrain: Only train models for stocks without existing models
    """
    try:
        st.title("Stock Anomaly Detection System")
        logger.info("Starting Streamlit application")

        uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                stocks = [col for col in data.columns if col != "Date"]

                # Create mlruns directory if it doesn't exist
                os.makedirs("mlruns", exist_ok=True)

                try:
                    logger.info("Initializing MLflow manager")
                    mlflow_manager = MLFlowManager("config/config.yaml")
                    logger.info("MLflow manager initialized successfully")

                    logger.info("Initializing model trainer")
                    trainer = ModelTrainer(
                        "config/config.yaml", "config/hyperparameters.yaml"
                    )
                    logger.info("Model trainer initialized successfully")

                    # Initialize results containers
                    all_results = {}
                    all_anomalies = []

                    # Create columns for status and progress
                    status_container = st.empty()
                    progress_bar = st.progress(0)

                    # Create two columns for buttons
                    col1, col2 = st.columns(2)

                    # Force retrain button in first column
                    if col1.button("Force Retrain All Models"):
                        st.warning(
                            "Force retraining all models regardless of existing models..."
                        )
                        all_anomalies = []
                        all_results = {}

                        # Create a placeholder for the dynamic message
                        training_message = st.empty()

                        for idx, stock in enumerate(stocks):
                            # Update message to show remaining stocks
                            remaining_stocks = ", ".join(stocks[idx:])
                            training_message.warning(
                                f"Models remaining to train: {remaining_stocks}"
                            )

                            if (
                                idx < len(stocks) - 1
                            ):  # Don't show "retraining" message for last stock
                                status_container.text(
                                    f"Force retraining model for {stock}..."
                                )

                            progress_bar.progress((idx + 1) / len(stocks))

                            anomalies, predictions, scores = trainer.train_stock_model(
                                stock, data
                            )
                            if not anomalies.empty:
                                anomalies["is_anomaly"] = True
                                all_anomalies.append(anomalies)
                            all_results[stock] = (predictions, scores)

                            # Update completion message
                            if idx < len(stocks) - 1:
                                training_message.success(
                                    f"Completed {stock}. Training remaining models..."
                                )
                            else:
                                training_message.success(
                                    "All models trained successfully!"
                                )
                                status_container.empty()  # Clear the status message for the last stock

                        if all_anomalies:
                            final_anomalies = pd.concat(all_anomalies)
                            final_anomalies.to_csv("anomalies.csv", index=False)
                            st.success(
                                "All models force retrained and anomalies saved!"
                            )

                            # Create download button for anomalies file
                            with open("anomalies.csv", "rb") as f:
                                st.download_button(
                                    label="Download Anomalies CSV",
                                    data=f,
                                    file_name="anomalies.csv",
                                    mime="text/csv",
                                    help="Click to download the anomalies data",
                                )

                            # Create and show plots
                            create_and_show_plots(data, stocks, all_results)

                            # Show anomalies dataframe
                            st.dataframe(final_anomalies)

                    # Regular retrain button in second column
                    elif col2.button("Retrain Missing Models Only"):
                        all_anomalies = []
                        all_results = {}
                        stocks_to_train = []

                        # Create a placeholder for the dynamic message
                        training_message = st.empty()

                        # Initialize DataLoader
                        data_loader = DataLoader(None)

                        # First, get existing models and their results
                        for stock in stocks:
                            model = mlflow_manager.get_model(stock)
                            if model is None:
                                stocks_to_train.append(stock)
                            else:
                                # For existing models, get predictions and scores
                                training_data, _, csv_values, _ = (
                                    data_loader.prepare_stock_data(data, stock)
                                )
                                predictions = model.predict(training_data)
                                scores = model.score_samples(training_data)
                                all_results[stock] = (predictions, scores)

                                # Get anomalies for existing model
                                anomaly_indices = np.where(predictions == -1)[0]
                                if len(anomaly_indices) > 0:
                                    anomalies = pd.DataFrame(
                                        {
                                            "Date": data["Date"].iloc[anomaly_indices],
                                            f"{stock}_Value": csv_values[
                                                anomaly_indices
                                            ],
                                            f"{stock}_PctChange": training_data.iloc[
                                                anomaly_indices, 0
                                            ].values,
                                            f"{stock}_AnomalyScore": scores[
                                                anomaly_indices
                                            ],
                                            f"{stock}_IsAnomaly": True,
                                        }
                                    )
                                    all_anomalies.append(anomalies)

                        # Train missing models if any
                        if stocks_to_train:
                            st.warning(
                                f"Training models for: {', '.join(stocks_to_train)}"
                            )

                            for idx, stock in enumerate(stocks_to_train):
                                # Update message to show remaining stocks
                                remaining = ", ".join(stocks_to_train[idx:])
                                training_message.warning(
                                    f"Models remaining to train: {remaining}"
                                )

                                progress_bar.progress((idx + 1) / len(stocks_to_train))

                                # Train model and store results
                                anomalies, predictions, scores = (
                                    trainer.train_stock_model(stock, data)
                                )
                                if not anomalies.empty:
                                    anomalies["is_anomaly"] = True
                                    all_anomalies.append(anomalies)
                                all_results[stock] = (predictions, scores)

                                # Update completion message
                                if idx < len(stocks_to_train) - 1:
                                    training_message.success(
                                        f"Completed {stock}. Training remaining models..."
                                    )
                                else:
                                    training_message.success(
                                        "All missing models trained successfully!"
                                    )
                                    status_container.empty()

                        # Create and show plots for all stocks
                        create_and_show_plots(data, stocks, all_results)

                        # Handle anomalies if any were found
                        if all_anomalies:
                            final_anomalies = pd.concat(all_anomalies)
                            final_anomalies.to_csv("anomalies.csv", index=False)

                            if stocks_to_train:
                                st.success(
                                    "New models trained and all anomalies saved!"
                                )
                            else:
                                st.success("Anomalies from existing models saved!")

                            # Create download button for anomalies file
                            with open("anomalies.csv", "rb") as f:
                                st.download_button(
                                    label="Download Anomalies CSV",
                                    data=f,
                                    file_name="anomalies.csv",
                                    mime="text/csv",
                                    help="Click to download the anomalies data",
                                )

                            # Show anomalies dataframe
                            st.dataframe(final_anomalies)
                        else:
                            st.info("No anomalies detected in any models.")

                        if not stocks_to_train:
                            st.info("All models already exist. No retraining needed.")

                except Exception as e:
                    logger.error(f"Error processing uploaded file: {str(e)}")
                    st.error(f"Error processing uploaded file: {str(e)}")
            except Exception as e:
                logger.error(f"Error initializing MLflow or trainer: {str(e)}")
                st.error(f"Error initializing MLflow or trainer: {str(e)}")
                return

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")


def create_and_show_plots(
    data: pd.DataFrame, stocks: List[str], all_results: Dict
) -> None:
    """Create plots using original CSV values and mark anomalies"""
    if all_results:
        n_stocks = len(stocks)
        n_cols = 3
        n_rows = (n_stocks + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=stocks,
            horizontal_spacing=0.1,
            vertical_spacing=0.2,
        )

        for idx, stock in enumerate(stocks):
            predictions, scores = all_results[stock]
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Plot original CSV values
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data[stock],  # Use original CSV values
                    mode="lines",
                    name=f"{stock} Values",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Add anomaly points using original CSV values
            anomaly_indices = np.where(predictions == -1)[0]
            if len(anomaly_indices) > 0:
                # Add 1 to indices because predictions are based on pct_change (which has one less point)
                csv_indices = [i + 1 for i in anomaly_indices]

                fig.add_trace(
                    go.Scatter(
                        x=data["Date"].iloc[csv_indices],
                        y=data[stock].iloc[csv_indices],
                        mode="markers",
                        name=f"{stock} Anomalies",
                        marker=dict(color="red", size=10),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        # Update layout with larger size and better formatting
        fig.update_layout(
            height=400 * n_rows,
            width=1400,
            title_text="Stock Values and Anomalies",
            showlegend=False,
            title_x=0.5,
            margin=dict(l=50, r=50, t=100, b=50),
        )

        # Update all x-axes to show better date formatting
        for i in range(1, n_rows * n_cols + 1):
            fig.update_xaxes(
                tickangle=45, row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1
            )

        # Add a unique key using timestamp
        unique_key = f"anomaly_plots_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
        st.plotly_chart(fig, key=unique_key, use_container_width=True)


if __name__ == "__main__":
    main()
