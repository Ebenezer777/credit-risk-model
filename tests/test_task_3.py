import os
from src.data_processing import process_data

# -----------------------------
# 1️⃣ Build the path to the CSV
# -----------------------------

csv_filename = "data.csv"
csv_path = os.path.join(os.getcwd(), "data", "raw", csv_filename)

# Check if file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

# -----------------------------
# 2️⃣ Run the data processing pipeline
# -----------------------------
X, preprocessor, feature_df = process_data(csv_path)

# -----------------------------
# 3️⃣ Print outputs for verification
# -----------------------------
print("\n✅ Feature dataframe (customer-level) shape:", feature_df.shape)
print("Columns in feature dataframe:", feature_df.columns.tolist())
print("\n✅ Transformed X shape (after pipeline):", X.shape)

# Optional: print first 5 rows of features
print("\nFirst 5 rows of feature dataframe:")
print(feature_df.head())
