"""

THIS NOTEBOOK CAN ONLY BE RUN IN DATABRICKS UI
IN VSCODE WON'T WORK

"""

# Databricks notebook source
import mlflow
import pandas as pd
from databricks import feature_engineering
from databricks.feature_store import FeatureLookup
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from credit_default.utils import load_config

CONFIG_DATABRICKS = "../../project_config.yml"
config = load_config(CONFIG_DATABRICKS)
parameters = config["parameters"]


# Initialize Spark and feature engineering client
spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

# Define your columns
columns = [
    "Limit_bal",
    "Sex",
    "Education",
    "Marriage",
    "Age",
    "Pay_0",
    "Pay_2",
    "Pay_3",
    "Pay_4",
    "Pay_5",
    "Pay_6",
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

# First, let's create the feature table with original data
create_table_sql = f"""
CREATE OR REPLACE TABLE maven.default.balanced_features
(Id STRING NOT NULL,
 {', '.join([f'{col} DOUBLE' for col in columns])})
"""
spark.sql(create_table_sql)

# Add primary key and enable CDF
spark.sql("ALTER TABLE maven.default.balanced_features " "ADD CONSTRAINT balanced_features_pk PRIMARY KEY(Id);")
spark.sql("ALTER TABLE maven.default.balanced_features " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# Convert Spark DataFrame to Pandas for SMOTE
train_pdf = spark.table("maven.default.train_set").toPandas()

# Separate features and target
X = train_pdf[columns]
y = train_pdf["Default"]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Create IDs for all samples (original + synthetic)
total_samples = len(X_balanced)
all_ids = [f"id_{i}" for i in range(total_samples)]

# Create balanced DataFrame
balanced_df = pd.DataFrame(X_balanced, columns=columns)
balanced_df["Id"] = all_ids

# Convert back to Spark DataFrame and insert into feature table
balanced_spark_df = spark.createDataFrame(balanced_df)
balanced_spark_df.write.format("delta").mode("append").saveAsTable("maven.default.balanced_features")

# Now use create_training_set to create balanced training set
# Drop the original features that will be looked up from the feature store
train_set = spark.table("maven.default.train_set").drop(*columns)

# Add Id column to train_set if it doesn't exist
if "Id" not in train_set.columns:
    train_set = train_set.withColumn("Id", F.monotonically_increasing_id().cast("string"))

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

training_set = fe.create_training_set(
    df=train_set,
    label="Default",
    feature_lookups=[
        FeatureLookup(
            table_name="maven.default.balanced_features",
            feature_names=columns,
            lookup_key="Id",
        )
    ],
    exclude_columns=["update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()
test_set = spark.table("maven.default.test_set").toPandas()

# Split features and target
X_train = training_df[columns]
y_train = training_df["Default"]
X_test = test_set[columns]
y_test = test_set["Default"]

features_robust = [
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

# Setup preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[("robust_scaler", RobustScaler(), features_robust)],
    remainder="passthrough",
)

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LGBMClassifier(**parameters))])


# COMMAND ----------
# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/credit-fe")

with mlflow.start_run(tags={"branch": "mlflow"}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    auc_test = roc_auc_score(y_test, y_pred)

    print("Test AUC:", auc_test)

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("AUC", auc_test)

    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )


mlflow.register_model(model_uri=f"runs:/{run_id}/lightgbm-pipeline-model-fe", name="maven.default.credit_model_fe")
