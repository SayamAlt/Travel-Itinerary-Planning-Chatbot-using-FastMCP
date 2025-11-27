from amadeus import Client, ResponseError
from fastmcp import FastMCP
from dotenv import load_dotenv
import os, requests

load_dotenv()

AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")

mcp = FastMCP("flight")

amadeus = Client(
    client_id=AMADEUS_API_KEY,
    client_secret=AMADEUS_API_SECRET
)

def get_airport_code(location: str):
    """ Returns the most relevant airport code for a given location. Performs a ranking of the most relevant airports. """
    url = "https://sky-scrapper.p.rapidapi.com/api/v1/flights/searchAirport"

    headers = {
        "x-rapidapi-key": os.getenv("RAPID_API_KEY"),
        "x-rapidapi-host": "sky-scrapper.p.rapidapi.com"
    }
    
    params = {
        "query": location
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    try:
        data = response.json().get("data", [])
    except Exception:
        return None

    if not data:
        return None
    
    query = location.lower().strip()
    
    def airport_score(item):
        title = item.get("presentation", {}).get("title", "").lower()
        suggestion = item.get("presentation", {}).get("suggestionTitle", "").lower()
        
        score = 0
        
        if query in title:
            score += 5
            
        if query in suggestion:
            score += 5
            
        if item.get("navigation", {}).get("entityType") == "AIRPORT":
            score += 2
            
        return score
    
    best_airport = sorted(data, key=airport_score, reverse=True)[0]
    sky_id = best_airport.get("skyId")
    return sky_id

@mcp.tool()
def search_flights(origin: str, destination: str, departure_date: str, num_adults: int):
    """ Search for flights using the Amadeus Flight Offers API. """
    origin, destination = get_airport_code(origin), get_airport_code(destination)
    
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            adults=num_adults,
            max=5
        ) 
        offers = response.data
        
        # Summarize flight offers
        summarized_offers = []
        
        for offer in offers:
            itinerary = offer["itineraries"][0]["segments"]
            segment = itinerary[0]

            departure = segment["departure"]["iataCode"]
            arrival = segment["arrival"]["iataCode"]
            dep_time = segment["departure"]["at"]
            arr_time = segment["arrival"]["at"]
            carrier = segment["carrierCode"]
            duration = offer["itineraries"][0].get("duration")

            price = offer.get("price", {}).get("total")
            currency = offer.get("price", {}).get("currency")

            summarized_offers.append({
                "airline": carrier,
                "from": departure,
                "to": arrival,
                "departure_time": dep_time,
                "arrival_time": arr_time,
                "duration": duration,
                "total_price": price,
                "currency_code": currency
            })
        
        return {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "num_adults": num_adults,
            "num_results": len(summarized_offers),
            "flights": summarized_offers
        }
    except ResponseError as error:
        return {"error": str(error)}
    
@mcp.tool()
def get_cheapest_flight(origin: str, destination: str):
    """ Get the cheapest flight between two locations using the Amadeus Cheapest Flight API. """
    origin, destination = get_airport_code(origin), get_airport_code(destination)
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            max=1,
            sort="price"
        )
        offers = response.data
        
        if not offers:
            return {"message": "No flights found."}
        
        offer = offers[0]
        itinerary = offer["itineraries"][0]["segments"]
        segment = itinerary[0]

        departure = segment["departure"]["iataCode"]
        arrival = segment["arrival"]["iataCode"]
        dep_time = segment["departure"]["at"]
        arr_time = segment["arrival"]["at"]
        carrier = segment["carrierCode"]
        duration = offer["itineraries"][0].get("duration")

        price = offer.get("price", {}).get("total")
        currency = offer.get("price", {}).get("currency")

        return {
            "airline": carrier,
            "from": departure,
            "to": arrival,
            "departure_time": dep_time,
            "arrival_time": arr_time,
            "duration": duration,
            "total_price": price,
            "currency_code": currency
        }
    except ResponseError as error:
        return {"error": str(error)}
    
if __name__ == "__main__":
    mcp.run()