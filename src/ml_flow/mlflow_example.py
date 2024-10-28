import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession

# Load environment variables
load_dotenv()

FILEPATH = os.environ["FILEPATH"]
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Load the house prices dataset
df = spark.read.csv("/Volumes/maven/default/data/data.csv", header=True, inferSchema=True).toPandas()

# COMMAND ----------
df.head()

# COMMAND ----------

# import os

# import mlflow
# import mlflow.sklearn
# from databricks.connect import DatabricksSession
# from databricks.sdk.core import Config
# from dotenv import load_dotenv
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split

# # Load environment variables
# load_dotenv()


# PROFILE = os.environ["PROFILE"]
# CLUSTER_ID = os.environ["CLUSTER_ID"]

# config = Config(profile="PROFILE", cluster_id="CLUSTER_ID")

# spark = DatabricksSession.builder.sdkConfig(config).getOrCreate()


# # Initialize a Databricks session
# # spark = DatabricksSession.builder.profile("dbc-df5087bc-8b50").getOrCreate()

# mlflow.set_tracking_uri(f"databricks://{PROFILE}")
# mlflow.set_experiment(experiment_name="/Shared/house-price-basic")
# # Load the classification dataset
# df = spark.read.csv("/Volumes/maven/default/data/data.csv", header=True, inferSchema=True).toPandas()

# # Print the first few rows of the dataframe
# print(df.head())

# # Preprocess the data (Assuming the target column is named 'target' and others are features)
# X = df.drop(columns=["default.payment.next.month"])  # Replace 'target' with your target variable name
# y = df["default.payment.next.month"]  # Replace 'target' with your target variable name

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Start an MLflow experiment
# mlflow.start_run()

# try:
#     # Initialize and train the model
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)

#     # Log parameters
#     mlflow.log_param("model_type", "Logistic Regression")
#     mlflow.log_param("test_size", 0.2)

#     # Make predictions
#     predictions = model.predict(X_test)

#     # Calculate metrics
#     accuracy = accuracy_score(y_test, predictions)
#     conf_matrix = confusion_matrix(y_test, predictions)
#     class_report = classification_report(y_test, predictions, output_dict=True)

#     # Log metrics
#     mlflow.log_metric("accuracy", accuracy)

#     # # Log the model
#     # mlflow.sklearn.log_model(model, "logistic_regression_model")

#     print(f"Accuracy: {accuracy:.2f}")
#     print("Confusion Matrix:")
#     print(conf_matrix)
#     print("Classification Report:")
#     print(classification_report(y_test, predictions))

# except Exception as e:
#     print(f"An error occurred: {e}")

# finally:
#     # End the MLflow run
#     mlflow.end_run()
