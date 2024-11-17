from kafka import KafkaConsumer
import json
from prometheus_client import start_http_server, Gauge, generate_latest
from flask import Flask, Response
import time

# Prometheus metrics
weather_temperature = Gauge('weather_temperature', 'Temperature in Celsius', ['city'])
weather_wind_speed = Gauge('weather_wind_speed', 'Wind speed in km/h', ['city'])
weather_visibility = Gauge('weather_visibility', 'Visibility (km)', ['city'])
weather_pressure = Gauge('weather_pressure', 'Pressure (mpa)', ['city'])
weather_humidity = Gauge('weather_humidity', 'Humidity', ['city'])

# Kafka configuration
kafka_topic_weather = 'test'  # Make sure this matches your producer topic
kafka_bootstrap_servers = 'localhost:9092'

# Create Kafka consumer
weather_consumer = KafkaConsumer(
    kafka_topic_weather,
    bootstrap_servers=kafka_bootstrap_servers,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='weather-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Initialize Flask app
app = Flask(__name__)

# Start the Prometheus HTTP server
start_http_server(8000)  # Expose metrics at http://localhost:8000/metrics

@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Return the metrics as a response.
    """
    # Generate the latest metrics and return them
    return Response(generate_latest(), mimetype='text/plain')

@app.route('/query/<city>', methods=['GET'])
def query_metrics(city):
    """
    Query metrics for a specific city.
    """
    try:
        temperature = weather_temperature.labels(city=city).get()
        wind_speed = weather_wind_speed.labels(city=city).get()
        visibility = weather_visibility.labels(city=city).get()
        pressure = weather_pressure.labels(city=city).get()
        humidity = weather_humidity.labels(city=city).get()

        return {
            'city': city,
            'temperature': temperature,
            'wind_speed': wind_speed,
            'visibility': visibility,
            'pressure': pressure,
            'humidity': humidity
        }, 200
    except Exception as e:
        return {'error': str(e)}, 404

# Main loop to consume messages and update metrics
try:
    while True:
        for message in weather_consumer:
            weather_data = message.value

            # Extract data
            city = weather_data['city']
            temperature = weather_data['temperature']
            wind_speed = weather_data['wind_speed']
            visibility = weather_data['visibility']
            pressure = weather_data['pressure']
            humidity = weather_data['humidity']

            # Update Prometheus metrics
            weather_temperature.labels(city=city).set(temperature)
            weather_wind_speed.labels(city=city).set(wind_speed)
            weather_visibility.labels(city=city).set(visibility)
            weather_pressure.labels(city=city).set(pressure)
            weather_humidity.labels(city=city).set(humidity)
            print(f"Updated metrics for {city}: Temperature = {temperature}, Wind Speed = {wind_speed}")

        time.sleep(1)  # Optional: Add a small sleep to prevent busy waiting

except KeyboardInterrupt:
    print("Shutting down...")

finally:
    weather_consumer.close()
