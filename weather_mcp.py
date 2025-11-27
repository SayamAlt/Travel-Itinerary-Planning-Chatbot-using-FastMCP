from fastmcp import FastMCP
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os, requests
import streamlit as st

load_dotenv()

if "secrets" in st.secrets:
    OPENAI_API_KEY = st.secrets["secrets"]["OPENAI_API_KEY"]
    OPENWEATHER_API_KEY = st.secrets["secrets"]["OPENWEATHER_API_KEY"]
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required.")

if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY environment variable is required.")

mcp = FastMCP("weather")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=OPENAI_API_KEY)

@mcp.tool()
def convert_fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert temperature from Fahrenheit to Celsius."""
    celsius = (fahrenheit - 32) * 5.0/9.0
    return round(celsius, 2)

def get_geographical_coordinates(location: str):
    """ Get the geographical coordinates (latitude and longitude) for a given location using the OpenWeatherMap Geocoding API. """
    url = "https://api.openweathermap.org/geo/1.0/direct"
    
    params = {
        "q": location,
        "limit": 1,
        "appid": OPENWEATHER_API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if not data:
        raise ValueError("Location not found")
    
    return data[0]["lat"], data[0]["lon"]

def get_weather_overview(forecast_data):
    """Generate a concise, travel-oriented weather overview using the LLM."""

    prompt = f"""
    You are a professional travel assistant. The user is planning a trip and needs a 
    clear, helpful, traveler-focused weather overview.

    Here is the raw multi-day forecast data:
    {forecast_data}

    Produce a concise summary that:
    - Starts with the overall trend (warm/cool, rainy/sunny, stable/changing)
    - Highlights important travel-relevant factors:
        * temperature ranges (day/night)
        * chance of rain or storms
        * humidity and comfort
        * wind severity
    - Gives packing recommendations (e.g., light jacket, umbrella, sunscreen)
    - Mentions any travel risks (heat alerts, storms, heavy rain, snow)
    - Stays friendly, clear, and non-technical
    - Avoids repeating raw data
    - Avoids unnecessary details

    Always convert temperature values to Celsius before giving recommendations using the convert_fahrenheit_to_celsius tool.
    Keep it short, helpful, and actionable for a traveler.
    """

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
@mcp.tool()
def get_weather_forecast(location: str, num_days: int = 5):
    """ """
    url = "https://api.openweathermap.org/data/2.5/forecast"
    
    # Get the latitude and longitude for the location
    latitude, longitude = get_geographical_coordinates(location)
    
    query_params = {
        "lat": latitude,
        "lon": longitude,
        "appid": OPENWEATHER_API_KEY
    }
    
    response = requests.get(url=url, params=query_params)
    weather_forecast = response.json()["list"][:num_days]  # Get the first num_days forecast entries
    weather_overview = get_weather_overview(weather_forecast)
    return weather_overview

if __name__ == "__main__":
    mcp.run()