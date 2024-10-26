import os
from typing import Any, Tuple

from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_default.data_preprocessing import DataPreprocessor
from credit_default.utils import load_config, setup_logging

# Load environment variables
load_dotenv()

FILEPATH = os.environ["FILEPATH"]
CONFIG = os.environ["CONFIG"]
TRAINING_LOGS = os.environ["TRAINING_LOGS"]


class ModelTrainer:
    """
    A class for training and evaluating a LightGBM model for credit default prediction.

    Attributes:
        X (pd.DataFrame): The features DataFrame.
        y (pd.Series): The target Series.
        preprocessor (ColumnTransformer): The preprocessor for scaling.
        model (LGBMClassifier): The LightGBM classifier model.
        X_train_scaled (pd.DataFrame): Scaled training features.
        X_val_scaled (pd.DataFrame): Scaled validation features.
        X_test_scaled (pd.DataFrame): Scaled test features.
        y_train (pd.Series): The training target Series.
        y_val (pd.Series): The validation target Series.
        y_test (pd.Series): The test target Series.
    """

    def __init__(self, X: Any, y: Any, preprocessor: Any, learning_rate: float, random_state: int) -> None:
        """
        Initializes the ModelTrainer class.

        Args:
            X (pd.DataFrame): The features DataFrame.
            y (pd.Series): The target Series.
            preprocessor (ColumnTransformer): The preprocessor for scaling.
            learning_rate (float): Learning rate for the model.
            random_state (int): Random state for reproducibility.
        """
        self.X = X
        self.y = y
        self.preprocessor = preprocessor
        self.model = LGBMClassifier(objective="binary", learning_rate=learning_rate)

        # Initialize placeholders for scaled features and target variables
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.random_state = random_state

    def train(self) -> None:
        """
        Trains the model using SMOTE for balancing the training set and scales the features.
        """
        logger.info("Starting model training process")

        try:
            # Split the data into training, validation, and test sets
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.X, self.y, test_size=0.30, random_state=self.random_state, stratify=self.y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
            )

            # Store the target variables
            self.y_train = y_train
            self.y_val = y_val
            self.y_test = y_test

            logger.info("Shape of X: {}", self.X.shape)
            logger.info("Feature columns in X: {}", self.X.columns.tolist())

            # Scale the features
            self.X_train_scaled = self.preprocessor.fit_transform(X_train)
            self.X_val_scaled = self.preprocessor.transform(X_val)
            self.X_test_scaled = self.preprocessor.transform(X_test)

            logger.info("X_train_scaled shape: {}", self.X_train_scaled.shape)
            logger.info("y_train shape: {}", self.y_train.shape)
            logger.info("X_val_scaled shape: {}", self.X_val_scaled.shape)
            logger.info("y_val shape: {}", self.y_val.shape)

            # Apply SMOTE to the training set
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(self.X_train_scaled, self.y_train)

            # Fit the model
            self.model.fit(
                X_train_resampled,
                y_train_resampled,
                eval_metric="logloss",
                eval_set=[(self.X_val_scaled, self.y_val)],
            )

            logger.info("Model training completed")
        except Exception:
            # logger.error("An error occurred during model training: {}", e)
            import traceback

            logger.error("An error occurred during model training: {}", traceback.format_exc())

            raise

    def evaluate(self) -> Tuple[Tuple[float, Any, str], Tuple[float, Any, str]]:
        """
        Evaluates the model on the validation and test sets.

        Returns:
            tuple: AUC score, confusion matrix, and classification report for validation and test sets.
        """
        logger.info("Evaluating model on the validation set")

        try:
            y_pred_val = self.model.predict(self.X_val_scaled)
            y_pred_proba_val = self.model.predict_proba(self.X_val_scaled)[:, 1]

            # Calculate AUC score for validation set
            auc_val = roc_auc_score(self.y_val, y_pred_proba_val)
            conf_matrix_val = confusion_matrix(self.y_val, y_pred_val)
            class_report_val = classification_report(self.y_val, y_pred_val)

            logger.info("Model evaluation on validation set completed")

            # Evaluate on test set
            logger.info("Evaluating model on the test set")

            y_pred_test = self.model.predict(self.X_test_scaled)
            y_pred_proba_test = self.model.predict_proba(self.X_test_scaled)[:, 1]

            # Calculate AUC score for test set
            auc_test = roc_auc_score(self.y_test, y_pred_proba_test)
            conf_matrix_test = confusion_matrix(self.y_test, y_pred_test)
            class_report_test = classification_report(self.y_test, y_pred_test)

            logger.info("Model evaluation on test set completed")

            return (auc_val, conf_matrix_val, class_report_val), (auc_test, conf_matrix_test, class_report_test)
        except Exception as e:
            logger.error("An error occurred during model evaluation: {}", e)
            raise


if __name__ == "__main__":
    # Set up logging
    setup_logging(TRAINING_LOGS)

    try:
        # Load configuration from YAML file
        config = load_config(CONFIG)

        # Extract parameters from the config
        learning_rate = config.parameters["learning_rate"]
        random_state = config.parameters["random_state"]

        # Initialize DataPreprocessor
        preprocessor_instance = DataPreprocessor(FILEPATH, config)
        X, y, preprocessor = preprocessor_instance.get_processed_data()

        # Initialize ModelTrainer and train the model
        model_trainer = ModelTrainer(X, y, preprocessor, learning_rate, random_state)
        model_trainer.train()

        # Evaluate the model
        (auc_val, conf_matrix_val, class_report_val), (auc_test, conf_matrix_test, class_report_test) = (
            model_trainer.evaluate()
        )

        logger.info("Validation AUC: {}", auc_val)
        logger.info("\nValidation Confusion Matrix:\n {}", conf_matrix_val)
        logger.info("\nValidation Classification Report:\n {}", class_report_val)

        logger.info("\nTest AUC: {}", auc_test)
        logger.info("\nTest Confusion Matrix:\n {}", conf_matrix_test)
        logger.info("\nTest Classification Report:\n {}", class_report_test)

    except Exception as e:
        logger.error("An error occurred in the main script: {}", e)

    logger.info("Model training and evaluation script completed")


# import os
# from typing import Any, Tuple

# from dotenv import load_dotenv
# from imblearn.over_sampling import SMOTE
# from lightgbm import LGBMClassifier
# from loguru import logger
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# from sklearn.model_selection import train_test_split

# from credit_default.data_preprocessing import DataPreprocessor
# from credit_default.utils import load_config, setup_logging

# # Load environment variables
# load_dotenv()

# FILEPATH = os.environ["FILEPATH"]
# CONFIG = os.environ["CONFIG"]
# TRAINING_LOGS = os.environ["TRAINING_LOGS"]


# class ModelTrainer:
#     """
#     A class for training and evaluating a LightGBM model for credit default prediction.

#     Attributes:
#         X (pd.DataFrame): The features DataFrame.
#         y (pd.Series): The target Series.
#         preprocessor (ColumnTransformer): The preprocessor for scaling.
#         model (LGBMClassifier): The LightGBM classifier model.
#         X_train_scaled (pd.DataFrame): Scaled training features.
#         X_val_scaled (pd.DataFrame): Scaled validation features.
#         X_test_scaled (pd.DataFrame): Scaled test features.
#         y_train (pd.Series): The training target Series.
#         y_val (pd.Series): The validation target Series.
#         y_test (pd.Series): The test target Series.
#     """

#     def __init__(self, X: Any, y: Any, preprocessor: Any, learning_rate: float, random_state: int) -> None:
#         """
#         Initializes the ModelTrainer class.

#         Args:
#             X (pd.DataFrame): The features DataFrame.
#             y (pd.Series): The target Series.
#             preprocessor (ColumnTransformer): The preprocessor for scaling.
#         """
#         self.X = X
#         self.y = y
#         self.preprocessor = preprocessor
#         self.model = LGBMClassifier(objective="binary", learning_rate=learning_rate)

#         # Initialize placeholders for scaled features and target variables
#         self.X_train_scaled = None
#         self.X_val_scaled = None
#         self.X_test_scaled = None
#         self.y_train = None
#         self.y_val = None
#         self.y_test = None
#         self.random_state = random_state

#     def train(self) -> None:
#         """
#         Trains the model using SMOTE for balancing the training set and scales the features.
#         """
#         logger.info("Starting model training process")

#         try:
#             # Split the data into training, validation, and test sets
#             X_train, X_temp, y_train, y_temp = train_test_split(
#                 self.X, self.y, test_size=0.30, random_state=self.random_state, stratify=self.y
#             )
#             X_val, X_test, y_val, y_test = train_test_split(
#                 X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
#             )

#             # Store the target variables
#             self.y_train = y_train
#             self.y_val = y_val
#             self.y_test = y_test

#             # Scale the features
#             self.X_train_scaled = self.preprocessor.fit_transform(X_train)
#             self.X_val_scaled = self.preprocessor.transform(X_val)
#             self.X_test_scaled = self.preprocessor.transform(X_test)

#             # Apply SMOTE to the training set
#             smote = SMOTE(random_state=42)
#             X_train_resampled, y_train_resampled = smote.fit_resample(self.X_train_scaled, self.y_train)

#             # Fit the model
#             self.model.fit(
#                 X_train_resampled,
#                 y_train_resampled,
#                 eval_metric="logloss",
#                 eval_set=[(self.X_val_scaled, self.y_val)],
#             )

#             logger.info("Model training completed")
#         except Exception as e:
#             logger.error("An error occurred during model training: {}", e)
#             raise

#     def evaluate(
#         self, X_val: Any, y_val: Any, X_test: Any, y_test: Any
#     ) -> Tuple[Tuple[float, Any, str], Tuple[float, Any, str]]:
#         """
#         Evaluates the model on the validation and test sets.

#         Args:
#             X_val (pd.DataFrame): The validation features DataFrame.
#             y_val (pd.Series): The validation target Series.
#             X_test (pd.DataFrame): The test features DataFrame.
#             y_test (pd.Series): The test target Series.

#         Returns:
#             tuple: AUC score, confusion matrix, and classification report for validation.
#         """
#         logger.info("Evaluating model on the validation set")

#         try:
#             y_pred_val = self.model.predict(X_val)
#             y_pred_proba_val = self.model.predict_proba(X_val)[:, 1]

#             # Calculate AUC score for validation set
#             auc_val = roc_auc_score(y_val, y_pred_proba_val)
#             conf_matrix_val = confusion_matrix(y_val, y_pred_val)
#             class_report_val = classification_report(y_val, y_pred_val)

#             logger.info("Model evaluation on validation set completed")

#             # Evaluate on test set
#             logger.info("Evaluating model on the test set")

#             y_pred_test = self.model.predict(X_test)
#             y_pred_proba_test = self.model.predict_proba(X_test)[:, 1]

#             # Calculate AUC score for test set
#             auc_test = roc_auc_score(y_test, y_pred_proba_test)
#             conf_matrix_test = confusion_matrix(y_test, y_pred_test)
#             class_report_test = classification_report(y_test, y_pred_test)

#             logger.info("Model evaluation on test set completed")

#             return (auc_val, conf_matrix_val, class_report_val), (auc_test, conf_matrix_test, class_report_test)
#         except Exception as e:
#             logger.error("An error occurred during model evaluation: {}", e)
#             raise


# if __name__ == "__main__":
#     # Set up logging
#     setup_logging(TRAINING_LOGS)

#     try:
#         # Load configuration from YAML file
#         config = load_config(CONFIG)

#         # Extract parameters from the config
#         learning_rate = config["parameters"]["learning_rate"]
#         random_state = config["parameters"]["random_state"]

#         # Initialize DataPreprocessor
#         preprocessor_instance = DataPreprocessor(FILEPATH, config)
#         X, y, preprocessor = preprocessor_instance.get_processed_data()

#         # Initialize ModelTrainer and train the model
#         model_trainer = ModelTrainer(X, y, preprocessor, learning_rate, random_state)
#         model_trainer.train()

#         # Evaluate the model
#         (auc_val, conf_matrix_val, class_report_val), (auc_test, conf_matrix_test, class_report_test) = (
#             model_trainer.evaluate(
#                 model_trainer.X_val_scaled, model_trainer.y_val, model_trainer.X_test_scaled, model_trainer.y_test
#             )
#         )

#         logger.info("Validation AUC: {}", auc_val)
#         logger.info("\nValidation Confusion Matrix:\n {}", conf_matrix_val)
#         logger.info("\nValidation Classification Report:\n {}", class_report_val)

#         logger.info("\nTest AUC: {}", auc_test)
#         logger.info("\nTest Confusion Matrix:\n {}", conf_matrix_test)
#         logger.info("\nTest Classification Report:\n {}", class_report_test)

#     except Exception as e:
#         logger.error("An error occurred in the main script: {}", e)

#     logger.info("Model training and evaluation script completed")
