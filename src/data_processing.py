# src/data_processing.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def impute_missing_values(df, num_cols=None, cat_cols=None):
    """Impute missing values for numerical and categorical columns."""
    df_copy = df.copy()
    if num_cols:
        for col in num_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    if cat_cols:
        for col in cat_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna("Unknown")
    return df_copy

def plot_numerical_distributions(df, num_cols):
    """Plot histograms and KDE for numerical columns."""
    for col in num_cols:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], bins=50, kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

def plot_boxplots(df, num_cols):
    """Plot boxplots for numerical columns."""
    for col in num_cols:
        if col in df.columns:
            plt.figure(figsize=(6, 3))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

def plot_categorical_distributions(df, cat_cols, top_n=10):
    """Plot bar plots for top N categories of categorical columns."""
    for col in cat_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            df[col].value_counts().head(top_n).plot(kind="bar")
            plt.title(f"Top Categories in {col}")
            plt.show()

def correlation_heatmap(df, num_cols):
    """Plot correlation heatmap for numerical columns."""
    if all(col in df.columns for col in num_cols):
        plt.figure(figsize=(8, 6))
        corr_matrix = df[num_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of Numerical Features")
        plt.show()
