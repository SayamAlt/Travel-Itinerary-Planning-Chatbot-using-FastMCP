from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import asyncio, threading, aiosqlite, os, logging
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
def build_itinerary(destination: str, days: int = 10, budget: float = 1000.0):
    """Build a travel itinerary for a given destination, duration, and budget."""

    system_prompt = f"""
            You are a Travel Orchestration Agent. Your job is to call MCP tools to gather 
            *real* information and then build an itinerary. You must follow ALL rules strictly:

            RULES

            1. **DO NOT answer directly from your own knowledge.**
            2. **DO NOT guess or invent flights, hotels, weather, or places.**
            3. **DO NOT skip tools. ALL tools must be used, in this exact order:**

            STEP 1 → After receiving flights → Call: `search_hotels(destination="{destination}", budget={budget})`
            STEP 2 → After receiving hotels → Call: `get_weather_forecast(location="{destination}")`
            STEP 3 → After receiving weather → Call: `search_tourism_destinations(city="{destination}")`

            4. Only after all 3 tool calls are complete, generate a final itinerary.

            OUTPUT FORMAT

            After collecting tool results, summarize everything using this structure:

            • **Destination:** {destination}  
            • **Trip Duration:** {days} days  
            • **Estimated Budget:** ${budget}

            ### Hotels (from tool)
            - Show recommended stays within budget.

            ### Weather Forecast (from tool)
            - Provide a 3–5 day summary.

            ### Places to Visit (from tool)
            - List 5–7 must-visit attractions.

            ### Day-by-Day Plan
            Create a {days}-day itinerary combining:
            - One major attraction per day  
            - Hotel suggestion  
            - Weather adjustment  
            - Optional local tips or food suggestions (optional, NOT fabricated)
    """
    response = llm_with_tools.invoke(system_prompt)
    return response

# Aggregate tools and bind to LLM
tools = [search_tool, build_itinerary, *mcp_tools]
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