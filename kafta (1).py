from kafka import KafkaProducer
import json
import requests
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Kafka setup
producer = KafkaProducer(bootstrap_servers='localhost:9092')

API_KEY = 'b306adcac73d9dfaab14e345187605b8'  # Your Weatherstack API key
city = 'Cape Town'
BASE_URL = 'http://api.weatherstack.com/current'

def get_weather_data(city):
    params = {
        'access_key': API_KEY,  # Use the Weatherstack API key
        'query': city + ',ZA'  # ZA is the country code for South Africa
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching weather data for {city}: {e}")
        return None


# Fetch Cape Town weather data in intervals
while True:
    weather_data = get_weather_data(city)
    if weather_data:  # Check if data was fetched successfully
        # Extract relevant fields from the response
        extracted_data = {
            'city': city,
            'local_time': weather_data['location']['localtime'],
            'temperature': weather_data['current']['temperature'],
            'wind_speed': weather_data['current']['wind_speed'],
            'wind_direction': weather_data['current']['wind_dir'],
            'visibility': weather_data['current']['visibility'],
            'humidity': weather_data['current']['humidity'],
            'weather_description': weather_data['current']['weather_descriptions'],
            'pressure': weather_data['current']['pressure'],
            'cloud_cover': weather_data['current']['cloudcover'],
            'feels_like': weather_data['current']['feelslike'],
        }

        # Send the extracted weather data to Kafka topic 'weather_me'
        producer.send('test', json.dumps(extracted_data).encode('utf-8'))
        print(f"Sent weather data for {city}: {extracted_data}")

    time.sleep(60)  # Wait for 60 seconds before fetching the next data
