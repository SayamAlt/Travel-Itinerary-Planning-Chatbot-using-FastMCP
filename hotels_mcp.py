from fastmcp import FastMCP
import os, requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st

load_dotenv()

if "secrets" in st.secrets:
    RAPID_API_KEY = st.secrets["secrets"]["RAPID_API_KEY"]
    RAPID_API_HOST = st.secrets["secrets"]["RAPID_API_HOST"]
    OPENAI_API_KEY = st.secrets["secrets"]["OPENAI_API_KEY"]
else:
    RAPID_API_KEY = os.getenv("RAPID_API_KEY")
    RAPID_API_HOST = "booking-com.p.rapidapi.com"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required.")

if not RAPID_API_KEY:
    raise ValueError("RAPID_API_KEY environment variable is required.")

if not RAPID_API_HOST:
    raise ValueError("RAPID_API_HOST environment variable is required.")

mcp = FastMCP("hotel")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=OPENAI_API_KEY)
    
def find_destination_id(location: str) -> str:
    """ Find the destination ID for a given location using the Booking.com API. """
    url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination"
    
    headers = {
        "x-rapidapi-host": RAPID_API_HOST,
        "x-rapidapi-key": RAPID_API_KEY
    }
    
    params = {
        "query": location
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    if not data.get("data"):
        raise ValueError("Destination not found")

    return data["data"][0]["dest_id"]

def get_destination_id(location: str, locale: str = "en-us"):
    """ Get the destination ID for a given location using the Booking.com Locations API. """
    url = "https://booking-com.p.rapidapi.com/v1/hotels/locations"
    
    headers = {
        "x-rapidapi-key": RAPID_API_KEY,
        "x-rapidapi-host": RAPID_API_HOST
    }
    
    response = requests.get(url=url, headers=headers, params={"name": location, "locale": locale})
    return response.json()[0]["dest_id"]

def extract_hotel_data(api_response, limit=5):
    """Extract and simplify hotel data from Booking.com API response."""
    hotels_raw = api_response.get("result", [])
    simplified = []

    for h in hotels_raw[:limit]:
        gross_info = h.get("composite_price_breakdown", {}).get("gross_amount_hotel_currency", {})
        
        price_info = gross_info.get("value")
        currency = gross_info.get("currency")
        
        simplified.append({
            "name": h.get("hotel_name") or h.get("hotel_name_trans"),
            "rating": h.get("review_score"),
            "review_word": h.get("review_score_word", ""),
            "price": round(float(price_info), 2) if price_info else "N/A",
            "currency": currency,
            "distance_to_center": h.get("distance_to_cc_formatted") or h.get("distance"),
            "free_cancellation": bool(h.get("is_free_cancellable", 0)),
            "address": h.get("address_trans") or "",
            "city": h.get("city") or h.get("city_trans") or ""
        })

    return simplified

def generate_hotel_recommendation(location: str, hotels: list):
    """ Generate a hotel recommendation overview using the LLM. """
    hotel_text = ""
    for h in hotels:
        hotel_text += (
            f"Hotel Name: {h['name']}\n"
            f"Rating: {h['rating']}\n"
            f"Price: {h['price']} {h['currency']}\n"
            f"Distance to Center: {h['distance_to_center']}\n"
            f"Free Cancellation: {h['free_cancellation']}\n"
            f"Address: {h['address']}\n"
            f"Hotel URL: {h['url']}\n"
            f"Image URL: {h['image_url']}\n\n"
        )
        
    prompt = f"""
        Act as a professional travel planning assistant.

        The user asked for hotel options in: {location}.

        Here is the raw structured hotel data:

        {hotel_text}

        Write a clean, friendly, concise, well-organized overview:
        - Start with a short summary
        - Then present 3–5 hotels as bullet points or sections
        - Highlight rating, price, cancellation, distance, and unique features
        - Include a link for each hotel - only if found and relevant, otherwise omit it
        - Avoid technical jargon
        - Use clear formatting and emojis sparingly
    """
    
    response = llm.invoke(prompt)
    return response.content
    
@mcp.tool()
def search_hotels(num_adults: int, num_children: int, checkin_date: str, checkout_date: str, location: str, children_ages: str = "5,0", units: str = "metric", destination_type: str = "city", order_by: str = "popularity", num_rooms: int = 1, currency_code: str = "USD", locale: str = "en-us", page_num: int = 0, categories_filter_ids: str = "class::2,class::4,free_cancellation::1"):
    """ Search for hotels using the Booking.com API and return a recommendation of hotels for a specific location. """
    url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
    destination_id = get_destination_id(location.title())
    
    headers = {
        "x-rapidapi-host": RAPID_API_HOST,
        "x-rapidapi-key": RAPID_API_KEY
    }
    
    queryString = {
        "adults_number": num_adults,
        "children_number": num_children,
        "units": units,
        "page_number": page_num,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "categories_filter_ids": categories_filter_ids,
        "children_ages": children_ages,
        "dest_type": destination_type,
        "dest_id": destination_id,
        "order_by": order_by,
        "room_number": num_rooms,
        "filter_by_currency": currency_code,
        "locale": locale
    }
    response = requests.get(url, headers=headers, params=queryString)
    hotels = extract_hotel_data(response.json(), limit=10)
    hotels = generate_hotel_recommendation(location, hotels)
    return hotels

if __name__ == "__main__":
    mcp.run()