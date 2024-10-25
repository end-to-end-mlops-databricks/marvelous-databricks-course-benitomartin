# Databricks notebook source

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Step 1: Load the existing table
df = spark.table("maven.default.train_set")

# Step 2: Create an empty DataFrame with the same schema
empty_df = df.limit(0)

# Step 3: Save the empty DataFrame as a new table
empty_df.write.saveAsTable("maven.default.train_set_smote")
