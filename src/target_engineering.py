# src/target_engineering.py
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# -----------------------------
# 1. Load raw data
# -----------------------------
raw_path = r"C:\Users\assef\Desktop\Kifiya AI Mastery\week4\credit-risk-model\data\raw\data.csv"
df = pd.read_csv(raw_path)

# -----------------------------
# 2. Calculate RFM
# -----------------------------
def calculate_rfm(df, snapshot_date=None):
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    
    scaler = StandardScaler()
    rfm[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    return rfm

rfm = calculate_rfm(df)

# -----------------------------
# 3. Cluster customers
# -----------------------------
def cluster_customers(rfm, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['Cluster'] = kmeans.fit_predict(rfm[['Recency', 'Frequency', 'Monetary']])
    return rfm

rfm = cluster_customers(rfm)

# -----------------------------
# 4. Assign high-risk
# -----------------------------
def assign_high_risk(rfm):
    cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_summary.sort_values(
        ['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]
    ).index[0]
    
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    return rfm

rfm = assign_high_risk(rfm)

# -----------------------------
# 5. Save processed CSV
# -----------------------------
processed_path = r"C:\Users\assef\Desktop\Kifiya AI Mastery\week4\credit-risk-model\data\processed"
os.makedirs(processed_path, exist_ok=True)

task4_csv_path = os.path.join(processed_path, "task4_features.csv")
rfm.to_csv(task4_csv_path, index=True)  # CustomerId is index
print(f"✅ Task 4 processed CSV saved to: {task4_csv_path}")
print("First 5 rows:")
print(rfm.head())
