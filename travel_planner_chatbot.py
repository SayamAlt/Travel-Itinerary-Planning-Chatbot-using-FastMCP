from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import asyncio, threading, aiosqlite, os, logging, requests, pytz
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from dotenv import load_dotenv
from typing import List
from datetime import datetime
import geopy.distance
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

load_dotenv()

logger = logging.getLogger("travel_planner_chatbot")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SERVERS = {
    "hotel": {
        "transport": "stdio",
        "command": "python3",
        "args": ["/Users/sayamkumar/Desktop/Data Science/Projects/Travel Planner Assistant/hotels_mcp.py"]
    },
    "flight": {
        "transport": "stdio",
        "command": "python3",
        "args": ["/Users/sayamkumar/Desktop/Data Science/Projects/Travel Planner Assistant/flights_mcp.py"]
    },
    "weather": {
        "transport": "stdio",
        "command": "python3",
        "args": ["/Users/sayamkumar/Desktop/Data Science/Projects/Travel Planner Assistant/weather_mcp.py"]
    },
    "places": {
        "transport": "stdio",
        "command": "python3",
        "args": ["/Users/sayamkumar/Desktop/Data Science/Projects/Travel Planner Assistant/places_mcp.py"]
    },
    "arithmetic": {
            "transport": "stdio",
            "command": "python3",
            "args": ["/Users/sayamkumar/Desktop/Data Science/MCP/math-mcp-server/main.py"]
    }
}

# Dedicated async loop thread for MCP client
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    """Schedule coroutine to backend event loop and return Future."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    """Run coroutine and wait for completion (synchronous call)."""
    return _submit_async(coro).result()

def submit_async_task(coro):
    """Schedule coroutine, return concurrent.futures.Future for later inspection."""
    return _submit_async(coro)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# Define search tool as fallback
search_tool = DuckDuckGoSearchRun(region="en-us")

# Define MCP client
mcp_client = MultiServerMCPClient(SERVERS)

def load_mcp_tools() -> List[BaseTool]:
    """Load tools from configured MCP servers. run_async because client.get_tools is async."""
    try:
        return run_async(mcp_client.get_tools())
    except Exception as e:
        # If nothing available, return empty list — system still works with local tools.
        print("Warning: failed to load MCP tools:", e)
        return []
    
mcp_tools = load_mcp_tools()
    
@tool
def exchange_currency(from_currency: str, to_currency: str, amount: float):
    """Converts an amount from one currency to another."""
    url = "https://currency-conversion-and-exchange-rates.p.rapidapi.com/convert"
    
    query_params = {
        "from": from_currency,
        "to": to_currency,
        "amount": amount
    }
    
    headers = {
        "x-rapidapi-host": "currency-conversion-and-exchange-rates.p.rapidapi.com",
        "x-rapidapi-key": os.getenv("RAPID_API_KEY")
    }
    
    response = requests.get(url, headers=headers, params=query_params)
    data = response.json()
    return data['result']

@tool
def convert_timezone(time_str: str, from_tz: str, to_tz: str):
    """ Converts a time from one timezone to another. """
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    
    from_zone = pytz.timezone(from_tz)
    to_zone = pytz.timezone(to_tz)

    converted = from_zone.localize(dt).astimezone(to_zone)
    return converted.strftime("%Y-%m-%d %H:%M")

@tool
def calculate_distance(city1: str, city2: str):
    """ Calculates the approximate distance between two cities in kilometers. """
    geolocator = Nominatim(user_agent="travel_planner")
    
    loc1 = geolocator.geocode(city1)
    loc2 = geolocator.geocode(city2)
    
    coords_1 = (loc1.latitude, loc1.longitude)
    coords_2 = (loc2.latitude, loc2.longitude)
    
    return geopy.distance.distance(coords_1, coords_2).km

@tool
def get_local_time(city: str):
    """ Returns the current local time in a city. """
    geolocator = Nominatim(user_agent="travel_planner")
    location = geolocator.geocode(city)
    
    if not location:
        return "Location not found"
    
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=location.longitude, lat=location.latitude)
    
    if not timezone_str:
        return "Timezone not found"
    
    tz = pytz.timezone(timezone_str)
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M")

@tool
def get_difference_in_timezones(location1: str, location2: str):
    """ Returns the difference between two times in hours. """
    time1, time2 = get_local_time(location1), get_local_time(location2)
    dt1 = datetime.strptime(time1, "%Y-%m-%d %H:%M")
    dt2 = datetime.strptime(time2, "%Y-%m-%d %H:%M")
    
    return (dt2 - dt1).total_seconds() / 3600
@tool
def convert_units(value: float, from_unit: str, to_unit: str):
    """Convert between distance, temperature (C/F), and weight units."""
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # Temperature
    if from_unit == "c" and to_unit == "f":
        return value * 9/5 + 32
    if from_unit == "f" and to_unit == "c":
        return (value - 32) * 5/9

    # Distance (km <-> miles)
    if from_unit == "km" and to_unit == "mi":
        return value * 0.621371
    if from_unit == "mi" and to_unit == "km":
        return value / 0.621371

    # Weight
    if from_unit == "kg" and to_unit == "lb":
        return value * 2.20462
    if from_unit == "lb" and to_unit == "kg":
        return value / 2.20462

    return f"Unsupported conversion {from_unit} → {to_unit}"

@tool
def build_itinerary(destination: str, days: int = 10, budget: float = 1000.0):
    """ Builds a travel itinerary for a given destination, duration, and budget using all MCP tools. """

    system_prompt = f"""
            You are a Travel Orchestration Agent. Your job is to call MCP tools to gather 
            *real* information and then build an itinerary. You must follow ALL rules strictly:

            RULES

            1. **DO NOT answer directly from your own knowledge.**
            2. **DO NOT guess or invent flights, hotels, weather, or places.**
            3. **DO NOT skip tools. ALL tools must be used in this exact order:**

            STEP 1 → Call: `search_flights(destination="{destination}")`
            STEP 2 → After receiving flights → Call: 
                     `search_hotels(destination="{destination}", budget={budget})`
            STEP 3 → After receiving hotels → Call: 
                     `get_weather_forecast(location="{destination}")`
            STEP 4 → After receiving weather → Call: 
                     `search_tourism_destinations(location="{destination}")`

            4. Only after all 4 tool calls are complete, generate the final itinerary.

            OUTPUT FORMAT

            After collecting tool results, summarize everything using this structure:

            • **Destination:** {destination}  
            • **Trip Duration:** {days} days  
            • **Estimated Budget:** ${budget}

            ### Flights (from tool)
            - Show 2–3 recommended real flights.

            ### Hotels (from tool)
            - Show recommended stays within budget.

            ### Weather Forecast (from tool)
            - Provide a 3–5 day summary.

            ### Places to Visit (from tool)
            - List 5–7 must-visit attractions.

            ### Day-by-Day Plan
            Create a {days}-day itinerary that includes:
            - One major attraction per day  
            - Hotel suggestion  
            - Weather adjustment  
            - Optional local tips or food suggestions (optional, NOT fabricated)
    """

    response = llm_with_tools.invoke(system_prompt)
    return response

@tool
def estimate_trip_cost(destination: str, days: int = 10, flight_cost: float = 500.0,
                       hotel_budget: float = 100.0, daily_expenses: float = 50.0):
    """
    Estimate total trip cost using MCP tools to fetch real hotel data,
    weather (optional context), and tourism spots (optional context).
    The purpose is to avoid hallucinating costs and always use tool data when available.
    """

    system_prompt = f"""
            You are a Travel Budget Estimation Agent. Your job is to call MCP tools to gather
            *real* information before generating any cost estimation. Follow these rules:

            RULES
            1. **DO NOT guess or estimate hotel prices or attractions on your own.**
            2. **DO NOT skip tools. Use tools in this strict order:**

            STEP 1 → Call: `search_hotels(destination="{destination}", budget={hotel_budget})`
            STEP 2 → Call: `get_weather_forecast(location="{destination}")` (context only)
            STEP 3 → Call: `search_tourism_destinations(location="{destination}")` (context only)

            3. After all tool calls are complete, calculate:
               • Flight cost: ${flight_cost}
               • Hotel cost: (use real hotel prices returned by the tool)
               • Daily expenses: {days} × ${daily_expenses}
               • Total cost = flight + hotel_total + daily_expenses_total

            OUTPUT FORMAT
            After gathering tool results, output:
            • Destination
            • Days
            • Flight Cost
            • Hotel Cost (use average of recommended stays)
            • Daily Expenses
            • Total Estimate
    """

    response = llm_with_tools.invoke(system_prompt)
    return response

@tool
def generate_packing_list(destination: str, days: int = 7, trip_type: str = "general"):
    """Generate a packing list using MCP tools (weather + places) before producing the final list."""

    system_prompt = f"""
            You are a Travel Packing List Orchestration Agent. You MUST follow these rules exactly.

            RULES

            1. **DO NOT answer from your own knowledge.**
            2. **DO NOT guess weather or activities.**
            3. **ALL tool calls must happen in this exact order:**

            STEP 1 → Call: `get_weather_forecast(location="{destination}")`
            STEP 2 → Call: `search_tourism_destinations(city="{destination}")`

            4. After collecting BOTH tool results, generate a final packing list.

            PACKING LIST REQUIREMENTS

            • Include general essentials (chargers, passport, toiletries).  
            • Add weather-dependent items (e.g., jacket, umbrella).  
            • Add activity-dependent items (e.g., hiking shoes, swimwear).  
            • Do NOT invent weather or places — rely ONLY on tool results.  

            OUTPUT FORMAT

            • **Destination:** {destination}  
            • **Trip Duration:** {days} days  
            • **Trip Type:** {trip_type}

            ### Weather Insights (from tool)

            ### Activity Insights (from tool)

            ### Packing List
            - 10–20 well-categorized items  
            - Include essentials + weather gear + activity gear  
            - No hallucinated weather or activities
    """

    response = llm_with_tools.invoke(system_prompt)
    return response

# Aggregate tools and bind to LLM
tools = [search_tool, build_itinerary, calculate_distance, get_difference_in_timezones,exchange_currency, convert_timezone, convert_units, estimate_trip_cost, generate_packing_list, get_local_time, *mcp_tools]
llm_with_tools = llm.bind_tools(tools, tool_choice="auto") if tools else llm

# Define chat state schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
   
async def chat_node(state: ChatState):
    """Chat node that processes messages and generates a response using the LLM with tools."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

# Define the tool node
tool_node = ToolNode(tools) if tools else None

# Define a checkpointer for saving chat history
checkpoint_db_path = "travel_planner_chatbot.db"

async def _init_checkpointer():
    connection = await aiosqlite.connect(checkpoint_db_path)
    return AsyncSqliteSaver(conn=connection)

checkpointer = run_async(_init_checkpointer())

# Define the state graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition) # If LLM invokes a tool, go to tool_node
    graph.add_edge("tools", "chat_node")  # After tool execution, return to chat_node
else:
    graph.add_edge("chat_node", END)  # Directly end if no tools available
    
# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)
    
# Helper utilitiess
async def _alist_threads():
    """List all saved thread ids from the checkpointer."""
    threads = set()
    async for checkpoint in checkpointer.alist(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(threads)

def retrieve_all_threads():
    return run_async(_alist_threads())