
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Weather Prediction with Multiple Linear Regression") \
    .getOrCreate()

# Define the HDFS path to the CSV file
hdfs_path = "hdfs://localhost:9000/user/kwind/weather_predictor/weatherHistory.csv"  # replace 'localhost' with your actual NameNode IP or hostname if needed

# Load data from Hadoop HDFS
data = spark.read.csv(hdfs_path, header=True, inferSchema=True, sep=',')

# Clean column names by stripping whitespace
data = data.toDF(*[col_name.strip() for col_name in data.columns])

# Show the schema of the dataset
data.printSchema()

# Print the actual column names to check for typos or trailing spaces
print(data.columns)

# Define the feature columns and label columns
feature_columns = ['Temperature (C)', 'Humidity', 'Pressure (millibars)']
visibility_label = 'Visibility (km)'  # Adjust this to match your actual label column
wind_speed_label = 'Wind Speed (km/h)'  # Adjust this to match your actual label column

# Create feature vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data_prepared = assembler.transform(data)

# Apply scaling to the feature vector
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
scaler_model = scaler.fit(data_prepared)
data_scaled = scaler_model.transform(data_prepared)

# Select relevant columns for visibility and wind speed predictions
data_visibility = data_scaled.select("scaledFeatures", visibility_label)
data_wind_speed = data_scaled.select("scaledFeatures", wind_speed_label)

# Split the data into training and testing sets for both visibility and wind speed
train_visibility, test_visibility = data_visibility.randomSplit([0.8, 0.2], seed=1234)
train_wind_speed, test_wind_speed = data_wind_speed.randomSplit([0.8, 0.2], seed=1234)

# Train a Linear Regression model for visibility prediction
lr_visibility = LinearRegression(featuresCol="scaledFeatures", labelCol=visibility_label, regParam=0.01)
lr_visibility_model = lr_visibility.fit(train_visibility)

# Train a Linear Regression model for wind speed prediction
lr_wind_speed = LinearRegression(featuresCol="scaledFeatures", labelCol=wind_speed_label, regParam=0.01)
lr_wind_speed_model = lr_wind_speed.fit(train_wind_speed)

# Print the coefficients and intercepts for each model
print(f"Visibility Model - Coefficients: {lr_visibility_model.coefficients}")
print(f"Visibility Model - Intercept: {lr_visibility_model.intercept}")
print(f"Wind Speed Model - Coefficients: {lr_wind_speed_model.coefficients}")
print(f"Wind Speed Model - Intercept: {lr_wind_speed_model.intercept}")

# Evaluate model performance for visibility prediction
visibility_predictions = lr_visibility_model.transform(test_visibility)
visibility_evaluator = RegressionEvaluator(labelCol=visibility_label, predictionCol="prediction", metricName="rmse")
visibility_rmse = visibility_evaluator.evaluate(visibility_predictions)
print(f"Visibility Prediction RMSE: {visibility_rmse}")

# Evaluate model performance for wind speed prediction
wind_speed_predictions = lr_wind_speed_model.transform(test_wind_speed)
wind_speed_evaluator = RegressionEvaluator(labelCol=wind_speed_label, predictionCol="prediction", metricName="rmse")
wind_speed_rmse = wind_speed_evaluator.evaluate(wind_speed_predictions)
print(f"Wind Speed Prediction RMSE: {wind_speed_rmse}")

# Plotting RMSE for both models
plt.figure(figsize=(10, 6))
models = ['Visibility Model', 'Wind Speed Model']
rmse_values = [visibility_rmse, wind_speed_rmse]
plt.bar(models, rmse_values, color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Model RMSE Comparison')
plt.show()

# Convert predictions to Pandas DataFrame for plotting
visibility_predictions_pd = visibility_predictions.select("scaledFeatures", visibility_label, "prediction").toPandas()
wind_speed_predictions_pd = wind_speed_predictions.select("scaledFeatures", wind_speed_label, "prediction").toPandas()

# Plot Actual vs Predicted for Visibility
plt.figure(figsize=(10, 6))
plt.scatter(visibility_predictions_pd[visibility_label], visibility_predictions_pd["prediction"], color='blue')
plt.plot([visibility_predictions_pd[visibility_label].min(), visibility_predictions_pd[visibility_label].max()],
         [visibility_predictions_pd[visibility_label].min(), visibility_predictions_pd[visibility_label].max()],
         '--', color='red')
plt.xlabel('Actual Visibility (km)')
plt.ylabel('Predicted Visibility (km)')
plt.title('Actual vs Predicted Visibility')
plt.show()

# Plot Actual vs Predicted for Wind Speed
plt.figure(figsize=(10, 6))
plt.scatter(wind_speed_predictions_pd[wind_speed_label], wind_speed_predictions_pd["prediction"], color='green')
plt.plot([wind_speed_predictions_pd[wind_speed_label].min(), wind_speed_predictions_pd[wind_speed_label].max()],
         [wind_speed_predictions_pd[wind_speed_label].min(), wind_speed_predictions_pd[wind_speed_label].max()],
         '--', color='red')
plt.xlabel('Actual Wind Speed (km/h)')
plt.ylabel('Predicted Wind Speed (km/h)')
plt.title('Actual vs Predicted Wind Speed')
plt.show()

# Save both models to HDFS
lr_visibility_model.write().overwrite().save("hdfs://localhost:9000/user/kwind/weather_predictor/Visibility_Model")
lr_wind_speed_model.write().overwrite().save("hdfs://localhost:9000/user/kwind/weather_predictor/WindSpeed_Model")

# Load the models for future use
from pyspark.ml.regression import LinearRegressionModel

loaded_visibility_model = LinearRegressionModel.load("hdfs://localhost:9000/user/kwind/weather_predictor/Visibility_Model")
loaded_wind_speed_model = LinearRegressionModel.load("hdfs://localhost:9000/user/kwind/weather_predictor/WindSpeed_Model")

# Make predictions on test data and display predictions for visibility
print("Visibility Predictions:")
visibility_predictions.select("scaledFeatures", visibility_label, "prediction").show(truncate=False)

# Show predictions for wind speed
print("Wind Speed Predictions:")
wind_speed_predictions.select("scaledFeatures", wind_speed_label, "prediction").show(truncate=False)

# Stop the Spark session
spark.stop()
