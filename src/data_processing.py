import pandas as pd
import numpy as np
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load raw transaction data from CSV"""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and datatype correction"""
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(
        df['TransactionStartTime'], errors='coerce'
    )
    return df


def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Customer-level numerical aggregates"""
    return (
        df.groupby('CustomerId')
        .agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            transaction_count=('TransactionId', 'count'),
            std_amount=('Amount', 'std')
        )
        .reset_index()
    )


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract transaction time components"""
    df = df.copy()
    df['tx_hour'] = df['TransactionStartTime'].dt.hour
    df['tx_day'] = df['TransactionStartTime'].dt.day
    df['tx_month'] = df['TransactionStartTime'].dt.month
    df['tx_year'] = df['TransactionStartTime'].dt.year
    return df


def aggregate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate time features per customer"""
    return (
        df.groupby('CustomerId')
        .agg(
            avg_tx_hour=('tx_hour', 'mean'),
            avg_tx_day=('tx_day', 'mean'),
            common_tx_month=('tx_month', lambda x: x.mode().iloc[0])
        )
        .reset_index()
    )


def extract_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dominant categorical behavior per customer"""
    return (
        df.groupby('CustomerId')
        .agg(
            channel=('ChannelId', lambda x: x.mode().iloc[0]),
            product_category=('ProductCategory', lambda x: x.mode().iloc[0]),
            provider=('ProviderId', lambda x: x.mode().iloc[0])
        )
        .reset_index()
    )


def merge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge all engineered features"""
    agg_features = create_aggregate_features(df)

    df_time = extract_time_features(df)
    time_features = aggregate_time_features(df_time)

    cat_features = extract_categorical_features(df)

    final_df = agg_features.merge(time_features, on='CustomerId', how='left')
    final_df = final_df.merge(cat_features, on='CustomerId', how='left')

    return final_df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values"""
    df = df.copy()

    for col in df.select_dtypes(include=np.number):
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def build_preprocessing_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """Sklearn preprocessing pipeline"""
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include='object').columns.tolist()

    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )


def process_data(path: str):
    """
    Task 3 Feature Engineering Pipeline
    Saves task3_features.csv for visualization
    """
    df = load_data(path)
    df = clean_data(df)

    feature_df = merge_features(df)
    feature_df = handle_missing(feature_df)

    # ✅ SAVE TASK 3 CSV
    processed_path = r"C:\Users\assef\Desktop\Kifiya AI Mastery\week4\credit-risk-model\data\processed"
    os.makedirs(processed_path, exist_ok=True)

    task3_csv_path = os.path.join(processed_path, "task3_features.csv")
    feature_df.to_csv(task3_csv_path, index=False)

    print(f"✅ Task 3 features saved to:\n{task3_csv_path}")

    preprocessor = build_preprocessing_pipeline(feature_df)
    X = preprocessor.fit_transform(feature_df)

    return X, preprocessor, feature_df


# ---------------------------------
# RUN SCRIPT DIRECTLY
# ---------------------------------
if __name__ == "__main__":
    RAW_DATA_PATH = r"C:\Users\assef\Desktop\Kifiya AI Mastery\week4\credit-risk-model\data\raw\data.csv"
    process_data(RAW_DATA_PATH)
