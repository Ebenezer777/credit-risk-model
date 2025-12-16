# tests/test_data_processing.py
import unittest
import pandas as pd
from src.data_processing import (
    load_data, impute_missing_values,
    plot_numerical_distributions, plot_boxplots,
    plot_categorical_distributions, correlation_heatmap
)

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """Create a small sample DataFrame for testing."""
        self.df = pd.DataFrame({
            "Amount": [100, 200, None, 400],
            "Value": [50, 150, 250, None],
            "ProductCategory": ["A", "B", None, "D"],
            "ChannelId": ["C1", "C2", "C1", "C3"]
        })

    def test_impute_missing_values(self):
        """Test that missing values are imputed correctly."""
        num_cols = ["Amount", "Value"]
        cat_cols = ["ProductCategory"]
        df_imputed = impute_missing_values(self.df, num_cols=num_cols, cat_cols=cat_cols)
        
        # Check that no NaN values remain in specified columns
        for col in num_cols + cat_cols:
            self.assertFalse(df_imputed[col].isna().any())

        # Check that numerical imputation used median
        self.assertEqual(df_imputed["Amount"][2], 200)  # median of [100, 200, 400]
        self.assertEqual(df_imputed["Value"][3], 150)   # median of [50, 150, 250]
        
        # Check that categorical imputation used "Unknown"
        self.assertEqual(df_imputed["ProductCategory"][2], "Unknown")

    def test_load_data_file_not_found(self):
        """Test loading a non-existent file returns None."""
        result = load_data("non_existent_file.csv")
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
