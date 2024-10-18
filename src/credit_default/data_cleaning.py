import pandas as pd
from loguru import logger


class DataCleaning:
    """A class for cleaning and preprocessing credit default data.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the loaded data.
        config (dict): The configuration dictionary containing preprocessing settings.
        X (pd.DataFrame or None): Features DataFrame after preprocessing, initialized to None.
        y (pd.Series or None): Target Series after preprocessing, initialized to None.
        preprocessor (sklearn.compose.ColumnTransformer or None): ColumnTransformer for preprocessing, initialized to None.
    """

    def __init__(self, filepath, config):
        """
        Initializes the DataCleaning class.

        Args:
            filepath (str): The path to the CSV file containing the data.
            config (dict): The configuration dictionary containing preprocessing settings.
        """
        self.df = self.load_data(filepath)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, filepath):
        """
        Loads data from a CSV file.

        Args:
            filepath (str): The path to the CSV file to be loaded.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.
        """
        logger.info("Loading data from {}", filepath)
        return pd.read_csv(filepath)

    def preprocess_data(self):
        """
        Preprocesses the data by performing several cleaning steps, including:
        - Removing specified columns.
        - Renaming the target column.
        - Capitalizing column names.
        - Correcting unknown values in specified columns.

        Returns:
            pd.DataFrame: The cleaned DataFrame after preprocessing.
        """
        logger.info("Starting data preprocessing")

        # Remove specified columns from the config
        columns_to_drop = self.config.get("columns_to_drop", [])
        logger.info("Dropping columns: {}", columns_to_drop)
        self.df = self.df.drop(columns=columns_to_drop)

        # Rename the target column
        target_name = self.config["target"][0]["name"]
        new_target_name = "Default"
        logger.info("Renaming target column '{}' to '{}'", target_name, new_target_name)
        self.df = self.df.rename(columns={target_name: new_target_name})

        # Capitalize Columns
        logger.info("Capitalizing column names")
        self.df.columns = self.df.columns.str.capitalize()

        # Correct Unknown Values Education/Marriage/Pay
        logger.info("Correcting unknown values for Education, Marriage and Pay")
        self.df["Education"] = self.df["Education"].replace({0: 4, 5: 4, 6: 4})
        self.df["Marriage"] = self.df["Marriage"].replace({0: 3})

        columns_to_replace = ["Pay_0", "Pay_2", "Pay_3", "Pay_4", "Pay_5", "Pay_6"]
        logger.info("Replacing values for columns: {}", columns_to_replace)
        self.df[columns_to_replace] = self.df[columns_to_replace].replace({-1: 0, -2: 0})

        logger.info("Data preprocessing completed")
        return self.df
