from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import SparkSession
import time
from prometheus_client import start_http_server, Gauge
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Define your email credentials and message content
subject = 'Welcome to Our Medical Diagnosis Enhancement Project!'
email_sender = 'codecraftgroupproject@gmail.com'
email_password = 'Use the app password here'  # Use the app password here
email_receiver = 'Replace with the actual recipient email'  # Replace with the actual recipient email


# SMTP server configuration
SMTP_SERVER = "smtp.gmail.com"

# Initialize Prometheus metrics for batch processing
visibility_prediction_metric = Gauge('visibility_prediction', 'Predicted visibility value')
wind_speed_prediction_metric = Gauge('wind_speed_prediction', 'Predicted wind speed value')

# Function to send email alerts
def send_email_alert(subject, body, receiver):
    try:
        msg = MIMEMultipart()
        msg['From'] = email_sender
        msg['To'] = receiver
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        # Use SMTP with starttls for secure connection on port 587
        with smtplib.SMTP(SMTP_SERVER, 587) as server:
            server.starttls()  # Secure the connection
            server.login(email_sender, email_password)
            server.sendmail(email_sender, receiver, msg.as_string())

        print(f"Alert email sent to {receiver}: {subject}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to perform prediction and send to Prometheus
def predict_and_expose_metrics():
    # Initialize Spark session
    spark = SparkSession.builder.appName("WeatherPrediction").getOrCreate()

    # Load the trained models from HDFS
    loaded_visibility_model = LinearRegressionModel.load(
        "hdfs://localhost:9000/user/Phatu/weather_predictor/Visibility_Model")
    loaded_wind_speed_model = LinearRegressionModel.load(
        "hdfs://localhost:9000/user/Phatu/weather_predictor/WindSpeed_Model")

    # Load the test data (assuming it's in Parquet format, adjust path as needed)
    test_data = spark.read.parquet("hdfs://localhost:9000/weather_data")

    # Preprocessing: assemble features and scale them
    feature_columns = ["temperature", "wind_speed", "humidity"]  # Adjust according to your actual feature columns
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_data = assembler.transform(test_data)

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    scaled_data = scaler.fit(assembled_data).transform(assembled_data)

    # Apply the models to the scaled data and loop over each prediction in the batch
    visibility_predictions = loaded_visibility_model.transform(scaled_data)
    wind_speed_predictions = loaded_wind_speed_model.transform(scaled_data)

    # Iterate over each row in the predictions to expose metrics for each entry in the batch
    for visibility_row, wind_speed_row in zip(visibility_predictions.collect(), wind_speed_predictions.collect()):
        visibility_pred_value = visibility_row["prediction"]
        wind_speed_pred_value = wind_speed_row["prediction"]

        # Expose each prediction to Prometheus
        visibility_prediction_metric.set(visibility_pred_value)
        wind_speed_prediction_metric.set(wind_speed_pred_value)

        # Check for specific dangerous values to send alerts
        if wind_speed_pred_value > 70 or visibility_pred_value < 2:  # Adjust thresholds as needed
            subject = "Weather Alert: Dangerous Conditions Detected"
            body = (f"Alert! Dangerous conditions detected.\n\n"
                    f"Wind Speed: {wind_speed_pred_value} km/h\n"
                    f"Visibility: {visibility_pred_value} km\n")
            send_email_alert(subject, body, email_receiver)

        # For example purposes, let's add a short sleep to avoid overwhelming Prometheus with updates
        time.sleep(60)  # Adjust as necessary for real-time requirements

    # Stop the Spark session
    spark.stop()

# Start Prometheus HTTP server on port 8000
start_http_server(8000)

# Run predictions every 60 minutes
while True:
    predict_and_expose_metrics()
    time.sleep(60 * 60)  # Wait for 60 minutes
