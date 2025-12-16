# Credit Risk Probability Model for Alternative Data

## Overview
This project implements an end-to-end credit risk probability model using alternative data from an eCommerce platform. The goal is to enable Bati Bank to provide a buy-now-pay-later service by predicting customer credit risk based on transactional behavior.

---

## Credit Scoring Business Understanding

### Basel II and Model Interpretability
The Basel II Capital Accord emphasizes risk-sensitive capital requirements, requiring financial institutions to quantify, document, and justify their credit risk measurement methodologies. This regulatory focus makes model interpretability and transparency critical. An interpretable and well-documented model allows the bank to explain risk decisions to regulators, auditors, and internal stakeholders, ensures compliance, and reduces model governance risk. Black-box models without clear explanations may achieve higher performance but introduce regulatory and operational risks.

### Proxy Target Variable Justification
The dataset does not contain a direct indicator of loan default. As a result, creating a proxy target variable is necessary to approximate customer credit risk. In this project, customer behavioral patterns—such as transaction recency, frequency, and monetary value—are used to infer disengagement, which is assumed to correlate with higher default risk. However, this approach introduces business risks, including misclassification of customers and potential bias, since disengagement does not always imply inability or unwillingness to repay. Therefore, predictions based on this proxy should be treated as probabilistic risk signals rather than definitive default outcomes.

### Model Complexity Trade-offs in a Regulated Environment
Simple and interpretable models such as Logistic Regression with Weight of Evidence (WoE) provide transparency, stability, and regulatory acceptance, making them suitable for credit scoring scorecards. However, they may fail to capture complex non-linear relationships in alternative data. More complex models like Gradient Boosting can deliver higher predictive performance but reduce explainability and increase governance complexity. In regulated financial contexts, the trade-off involves balancing predictive accuracy with explainability, auditability, and regulatory compliance.

---

## Project Structure

credit-risk-model/
├── .github/workflows/ci.yml # CI/CD workflow for testing and linting
├── data/ # Add to .gitignore
│ ├── raw/ # Raw CSV files
│ └── processed/ # Processed and cleaned data
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis notebook
├── src/
│ ├── init.py
│ ├── data_processing.py # Feature engineering & cleaning
│ ├── eda_functions.py # Reusable plotting/statistics functions
│ ├── train.py # Model training logic
│ ├── predict.py # Model inference logic
│ └── api/
│ ├── main.py # FastAPI application
│ └── pydantic_models.py # API request/response validation
├── tests/
│ ├── test_data_processing.py # Unit tests for data functions
│ └── test_train.py # Optional: model unit tests
├── Dockerfile # Containerization
├── docker-compose.yml # Docker Compose setup
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore # Ignore environment, data, etc.

---

## Setup Instructions

1. Clone the repository:

```bash
git clone <repository-url>
cd credit-risk-model

2. Create and activate a Python virtual environment (outside the project folder):
python -m venv ../venv
# Windows

source ../venv/bin/activate

3. Install dependencies:
pip install --upgrade pip
pip install -r requirements.txt

4. Place raw data CSV in data/raw/.

5. Launch Jupyter Notebook to explore the notebooks:
jupyter notebook
