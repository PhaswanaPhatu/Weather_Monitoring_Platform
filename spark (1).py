from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, ArrayType
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create a Spark session
spark = SparkSession.builder \
    .appName("Real-Time Weather Alerts and Visualization") \
    .config("spark.kafka.bootstrap.servers", "localhost:9092") \
    .getOrCreate()

# Define Kafka configuration
kafka_bootstrap_servers = "localhost:9092"
kafka_topic = "test"

# Define the city you want to filter
target_city = "Cape Town"

# Create a DataFrame that reads from the Kafka topic
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic) \
    .load()

# Define the schema for the weather data
weather_schema = StructType([
    StructField("city", StringType()),
    StructField("local_time", StringType()),
    StructField("temperature", FloatType()),
    StructField("wind_speed", FloatType()),
    StructField("wind_direction", StringType()),
    StructField("visibility", FloatType()),
    StructField("humidity", IntegerType()),
    StructField("weather_description", ArrayType(StringType())),
    StructField("pressure", FloatType()),
    StructField("cloud_cover", FloatType()),
    StructField("feels_like", FloatType())
])

# Parse the JSON and select the relevant fields, filtering for the target city
parsed_df = kafka_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), weather_schema).alias("weather_data")) \
    .filter(col("weather_data.city") == target_city) \
    .select("weather_data.*") \
    .withColumn("processing_time", current_timestamp())  # Add processing timestamp

# Define thresholds for extreme weather conditions
storm_wind_speed_threshold = 80  # km/h
high_temperature_threshold = 35  # °C
low_visibility_threshold = 1  # km
high_humidity_threshold = 90  # %
low_pressure_threshold = 980  # hPa


# Function to process alerts and store them
def process_alerts(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    # Generate alerts based on thresholds
    alerts = batch_df.filter(
        (col("wind_speed") > storm_wind_speed_threshold) |
        (col("temperature") > high_temperature_threshold) |
        (col("visibility") < low_visibility_threshold) |
        (col("humidity") > high_humidity_threshold) |
        (col("pressure") < low_pressure_threshold)
    )

    if alerts.count() > 0:
        # Log alerts
        logging.info(f"Batch {batch_id}: Severe weather alerts generated for {target_city}")

        # Store alerts in a separate table/directory with timestamp
        alerts.write \
            .format("parquet") \
            .mode("append") \
            .partitionBy("city") \
            .save("hdfs://localhost:9000/weather_alerts")

        # Additionally write to Kafka if needed
        alerts.selectExpr("to_json(struct(*)) AS value") \
            .write \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
            .option("topic", "weather_alerts") \
            .save()


# Function to store weather data
def store_weather_data(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    # Store the complete weather data
    try:
        # Partition by city and store as parquet
        batch_df.write \
            .format("parquet") \
            .mode("append") \
            .partitionBy("city") \
            .save("hdfs://localhost:9000/weather_data")

        logging.info(f"Batch {batch_id}: Successfully stored {batch_df.count()} records")
    except Exception as e:
        logging.error(f"Batch {batch_id}: Error storing data - {str(e)}")


# Define checkpoint locations
checkpoint_location_alerts = "hdfs://localhost:9000/checkpoints/alerts"
checkpoint_location_weather = "hdfs://localhost:9000/checkpoints/weather"

# Start the streaming queries
# Weather data storage stream
weather_query = parsed_df.writeStream \
    .foreachBatch(store_weather_data) \
    .option("checkpointLocation", checkpoint_location_weather) \
    .trigger(processingTime="1 minute") \
    .start()

# Alert processing stream
alert_query = parsed_df.writeStream \
    .foreachBatch(process_alerts) \
    .option("checkpointLocation", checkpoint_location_alerts) \
    .trigger(processingTime="1 minute") \
    .start()

# Optional: Add console output for debugging
debug_query = parsed_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

# Keep the application running
spark.streams.awaitAnyTermination()