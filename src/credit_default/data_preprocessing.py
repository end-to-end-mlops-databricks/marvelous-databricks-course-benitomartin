from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from src.credit_default.data_cleaning import DataCleaning


class DataPreprocessor:
    """A class for preprocessing credit default data, including scaling features.

    Attributes:
        data_cleaning (DataCleaning): An instance of the DataCleaning class used for data preprocessing.
        cleaned_data (pd.DataFrame): The cleaned DataFrame after preprocessing.
        features_robust (list): List of feature names for robust scaling.
        X (pd.DataFrame): Features DataFrame after cleaning.
        y (pd.Series): Target Series after cleaning.
        preprocessor (ColumnTransformer): ColumnTransformer for scaling the features.
    """

    def __init__(self, filepath, config):
        """
        Initializes the DataPreprocessor class.

        Args:
            filepath (str): The path to the CSV file containing the data.
            config (dict): The configuration dictionary containing preprocessing settings.
        """
        # Initialize DataCleaning to preprocess data
        self.data_cleaning = DataCleaning(filepath, config)
        self.cleaned_data = self.data_cleaning.preprocess_data()

        # Define robust features for scaling
        self.features_robust = [
            "Limit_bal",
            "Bill_amt1",
            "Bill_amt2",
            "Bill_amt3",
            "Bill_amt4",
            "Bill_amt5",
            "Bill_amt6",
            "Pay_amt1",
            "Pay_amt2",
            "Pay_amt3",
            "Pay_amt4",
            "Pay_amt5",
            "Pay_amt6",
        ]

        # Define features and target
        self.X = self.cleaned_data.drop(columns=["Default"])
        self.y = self.cleaned_data["Default"]

        # Set up the ColumnTransformer for scaling
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("robust_scaler", RobustScaler(), self.features_robust)  # Apply RobustScaler to the selected features
            ],
            remainder="passthrough",  # Keep other columns unchanged
        )

    def get_processed_data(self):
        """Retrieves the processed features, target, and preprocessor.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: The features DataFrame.
                - pd.Series: The target Series.
                - ColumnTransformer: The preprocessor for scaling.
        """
        return self.X, self.y, self.preprocessor
