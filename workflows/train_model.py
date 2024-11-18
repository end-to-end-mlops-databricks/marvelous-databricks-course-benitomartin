"""
This script trains a LightGBM model for credit default prediction with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM classifier
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups
- Outputs model URI for downstream tasks

"""

import argparse

import mlflow
import pandas as pd
from databricks import feature_engineering

# from databricks.connect import DatabricksSession
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score  # classification_report, confusion_matrix,
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from credit_default.utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
git_sha = args.git_sha
job_run_id = args.job_run_id


config_path = f"{root_path}/project_config.yml"
config = load_config(config_path)


workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()
# spark = DatabricksSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

# Extract configuration details
catalog_name = config.catalog_name
schema_name = config.schema_name
target = config.target[0].new_name
parameters = config.parameters
features_robust = config.features.robust
columns = config.features.clean
columns_wo_id = columns.copy()
columns_wo_id.remove("Id")


# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.features_balanced"


# Convert Train/Test Spark DataFrame to Pandas
train_pdf = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Separate features and target
X = train_pdf[columns_wo_id]
y = train_pdf[target]

# Apply SMOTE
smote = SMOTE(random_state=parameters["random_state"])
X_balanced, y_balanced = smote.fit_resample(X, y)

# Create balanced DataFrame using only the train_set
balanced_df = pd.DataFrame(X_balanced, columns=columns_wo_id)

# Identify the number of original samples
num_original_samples = len(train_pdf)
len_range = len(train_pdf) + len(test_set) + 1

# Retain original Ids for the real samples and create new Ids for synthetic samples
# Start with sum length train+test+1 to avoid conflicts with existing Ids
balanced_df["Id"] = train_pdf["Id"].values.tolist() + [
    str(i) for i in range(len_range, len_range + len(balanced_df) - num_original_samples)
]

# Convert back to Spark DataFrame and insert into feature table
balanced_spark_df = spark.createDataFrame(balanced_df)

# Cast columns in balanced_spark_df to match the schema of the Delta table
columns_to_cast = ["Sex", "Education", "Marriage", "Age", "Pay_0", "Pay_2", "Pay_3", "Pay_4", "Pay_5", "Pay_6"]

for column in columns_to_cast:
    balanced_spark_df = balanced_spark_df.withColumn(column, F.col(column).cast("double"))

balanced_spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.features_balanced")

# Now use create_training_set to create balanced training set
# Drop the original features that will be looked up from the feature store
# Define the list of columns you want to drop, including "Update_timestamp_utc"
columns_to_drop = columns_wo_id + ["Update_timestamp_utc"]

# Drop the specified columns from the train_set
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(*columns_to_drop)

# Feature Lookup

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=f"{catalog_name}.{schema_name}.features_balanced",
            feature_names=columns_wo_id,
            lookup_key="Id",
        )
    ],
    exclude_columns=["Update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()


# Split features and target (exclude 'Id' from features)
X_train = training_df[columns_wo_id]
y_train = training_df[target]
X_test = test_set[columns_wo_id]
y_test = test_set[target]


# Setup preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[("robust_scaler", RobustScaler(), features_robust)],
    remainder="passthrough",
)

# Create the pipeline with preprocessing and the LightGBM classifier
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))])


# Set up the experiment
mlflow.set_experiment(experiment_name="/Shared/credit-feature")

# Start an MLflow run to track the training process
with mlflow.start_run(tags={"branch": "bundles", "git_sha": f"{git_sha}", "job_run_id": job_run_id}) as run:
    run_id = run.info.run_id

    # Train the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    auc_test = roc_auc_score(y_test, y_pred)

    print("Test AUC:", auc_test)

    # Log parameters, metrics, and signature
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("AUC", auc_test)
    # signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Signature with input example
    input_example = X_train.iloc[:5]
    signature = infer_signature(
        model_input=input_example,
        model_output=pipeline.predict(input_example),  # y_pred
    )
    # Log model with feature engineering
    # We will register in next step, if model is better than the previous one
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
        input_example=input_example,
    )

model_uri = f"runs:/{run_id}/lightgbm-pipeline-model-fe"
dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)  # noqa: F821
