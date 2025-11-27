from fastmcp import FastMCP
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os, requests
import streamlit as st

load_dotenv()

if "secrets" in st.secrets:
    OPENAI_API_KEY = st.secrets["secrets"]["OPENAI_API_KEY"]
    OPENWEATHER_API_KEY = st.secrets["secrets"]["OPENWEATHER_API_KEY"]
    FOURSQUARE_API_KEY = st.secrets["secrets"]["FOURSQUARE_API_KEY"]
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
    
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required.")

if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY environment variable is required.")

if not FOURSQUARE_API_KEY:
    raise ValueError("FOURSQUARE_API_KEY environment variable is required.")

mcp = FastMCP("places")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=OPENAI_API_KEY)

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


def simplify_places(places: list, limit: int = 5):
    """ Extract relevant information from the places data fetched from the Foursquare API. """
    simplified = []

    for p in places[:limit]:
        simplified.append({
            "name": p.get("name"),
            "distance_m": p.get("distance"),
            "categories": [c.get("name") for c in p.get("categories", [])],
            "address": p.get("location", {}).get("formatted_address"),
            "rating": p.get("rating"),
            "description": p.get("description"),
        })

    return simplified

def get_tourism_recommendations(location: str, places_data: list):
    """ Generate tourism place recommendations using the LLM. """

    prompt = f"""
        You are a professional travel assistant. The user is planning a trip and needs a 
        clear, helpful list of top tourist places to visit.

        Here is the raw places data:
        {places_data}

        Produce a concise list that:
        - Starts with a short introduction about tourism in {location}
        - Lists the top places to visit with brief descriptions
        - Highlights unique features or must-see aspects of each place
        - Suggests the best time to visit each place if relevant
        - Stays friendly, clear, and non-technical
        - Avoids repeating raw data
        - Avoids unnecessary details

        Keep it short, helpful, and actionable for a traveler.
    """

    response = llm.invoke(prompt)
    return response.content
    
@mcp.tool()
def search_tourism_destinations(location: str, radius: int = 1000, sort: str = "POPULARITY", limit: int = 10):
    """ Search for top tourism destinations using the Foursquare Places API and return recommendations. """
    latitude, longitude = get_geographical_coordinates(location)
    url = "https://places-api.foursquare.com/places/search"

    headers = {
        "accept": "application/json",
        "X-Places-Api-Version": "2025-06-17",
        "authorization": "Bearer " + str(FOURSQUARE_API_KEY)
    }
        
    query_params = {
        "limit": limit,
        "sort": sort,
        "radius": radius,
        "ll": str(latitude) + "," + str(longitude),
        "categories": "16000,13065,13032"
    }

    response = requests.get(url, headers=headers, params=query_params)
    tourism_places = response.json().get("results", [])
    tourism_places = simplify_places(tourism_places, limit=limit)
    tourism_suggestions = get_tourism_recommendations(location, tourism_places)
    return tourism_suggestions

if __name__ == "__main__":
    mcp.run()