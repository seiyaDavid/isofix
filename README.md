# Stock Anomaly Detection System

## Overview
This project implements an automated anomaly detection system for multiple stock time series data using Isolation Forest algorithm. The system features automated hyperparameter tuning with Optuna, model versioning with MLflow, and a Streamlit web interface for interactive analysis.

## Features
- Anomaly detection using Isolation Forest algorithm
- Automated hyperparameter optimization using Optuna
- Model versioning and tracking with MLflow
- Interactive web interface with Streamlit
- Dynamic visualization of stock data and anomalies
- Automated model retraining capabilities
- Support for multiple stocks simultaneously
- Progress tracking for model training
- Configurable hyperparameters via YAML files

## Project Structure 

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd anomaly_detection
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Configure hyperparameters in `config/hyperparameters.yaml`:
```yaml
isolation_forest:
  n_estimators:
    low: 50
    high: 300
  max_samples:
    low: 0.1
    high: 1.0
  contamination:
    low: 0.01
    high: 0.1
  max_features:
    low: 0.1
    high: 1.0
  bootstrap:
    options: [true, false]
```

2. Set up general configuration in `config/config.yaml`:
```yaml
mlflow:
  tracking_uri: "mlruns"
  experiment_name: "stock_anomaly_detection"

data:
  output_file: "anomalies.csv"
```

## Running the Application

1. Generate sample data (optional):
```bash
python scripts/generate_sample_data.py
```

2. Start MLflow tracking server (in a separate terminal):
```bash
mlflow ui
```
Access MLflow UI at: http://localhost:5000

3. Launch the Streamlit application:
```bash
streamlit run app.py
```
Access the web interface at: http://localhost:8501

## Using the Application

1. Upload Data:
   - Prepare your CSV file with columns: 'Date' and stock symbols
   - Upload the file using the file uploader in the web interface

2. Train Models:
   - Use "Force Retrain All Models" to retrain all stock models
   - Use "Retrain Missing Models Only" to train only for stocks without existing models

3. View Results:
   - Interactive plots showing stock values and detected anomalies
   - Anomaly details in tabular format
   - Exported results in 'anomalies.csv'

## Input Data Format
The system expects a CSV file with the following format:
```csv
Date,STOCK_1,STOCK_2,STOCK_3,...
2023-01-01,0.5,-0.2,0.3,...
2023-01-02,-0.1,0.4,0.2,...
```
- First column must be named 'Date'
- Subsequent columns should be stock symbols
- Values should be daily percentage changes

## Logging
- Application logs are stored in the `logs/` directory
- Each component has its own log file for easier debugging
- MLflow tracking information is stored in `mlruns/`

## Model Management
- Trained models are versioned and stored using MLflow
- Access model history and metrics through MLflow UI
- Models can be reused or retrained as needed

## Contributing
[Add contribution guidelines if applicable]

## License
[Add license information]

## Contact
[Add contact information]

## Testing

1. **Run Tests**:
```bash
# From the project root directory
pytest tests/ -v

# For tests with coverage report
pytest tests/ -v --cov=.
```

2. **Run API**:
```bash
# Method 1: Using uvicorn directly
uvicorn api:app --reload

# Method 2: Using python
python api.py
```

The API will be available at:
- Main API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Alternative API Documentation: http://localhost:8000/redoc

3. **Test API Endpoints** using curl:
```bash
# Test detect_anomalies endpoint
curl -X POST "http://localhost:8000/detect_anomalies/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_data/stock_data.csv"

# Test force_retrain endpoint
curl -X POST "http://localhost:8000/force_retrain/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_data/stock_data.csv"
```

4. **Generate Sample Data** (if needed):
```bash
# Generate sample data first
python scripts/generate_sample_data.py
```

Directory Structure Should Be:
```
your_project/
├── config/
│   ├── config.yaml
│   └── hyperparameters.yaml
├── sample_data/
│   └── stock_data.csv
├── tests/
├── src/
├── api.py
└── requirements.txt
```

Remember to:
1. Have all dependencies installed
2. Be in the project root directory
3. Have sample data available
4. Have config files in place 

