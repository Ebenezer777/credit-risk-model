import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import process_data
from src.target_engineering import calculate_rfm, cluster_customers, assign_high_risk, merge_target

# Load processed features
csv_path = os.path.join(os.getcwd(), "data", "raw", "data.csv")
X, preprocessor, feature_df = process_data(csv_path)

# Task 4: RFM + Clustering + High-risk target
rfm = calculate_rfm(pd.read_csv(csv_path))
rfm = cluster_customers(rfm)
rfm = assign_high_risk(rfm)

feature_df = merge_target(feature_df, rfm)

# Verification
print("Feature dataframe with target shape:", feature_df.shape)
print("Columns:", feature_df.columns.tolist())
print("High-risk distribution:\n", feature_df['is_high_risk'].value_counts())

