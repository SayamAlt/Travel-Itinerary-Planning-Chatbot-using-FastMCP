import streamlit as st
from travel_planner_chatbot import chatbot, retrieve_all_threads, submit_async_task
import queue
from uuid import uuid4
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

def generate_thread_id():
    return str(uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
        
def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values
    return state.get("messages", [])

# Initialize session state
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
    
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
    
add_thread(st.session_state["thread_id"])

# Create sidebar for thread management
st.set_page_config(page_title="✈️ Travel Planning Assistant", page_icon="✈️")
st.title("✈️ Travel Planning Assistant")

# Define a new chat button to reset the chat
if st.sidebar.button("New Chat"):
    reset_chat()
    
st.sidebar.header("Conversations")

# List existing chat threads
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temporary_messages = []
        
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temporary_messages.append({"role": role, "content": msg.content})
            
        st.session_state["message_history"] = temporary_messages

# Display chat messages from history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])
        
# Accept user input
user_input = st.chat_input("Ask me about your travel plans...")

if user_input:
    # Show user message in chat
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.text(user_input)
        
    # Prepare chat config with thread id
    chat_config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn"
    }
    
    # Submit async task to chatbot
    with st.chat_message("assistant"):
        status_holder = {"box": None, "used_tools": []}  # For tool progress display
        
        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=chat_config,
                        stream_mode="messages"
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as e:
                    event_queue.put(("error", e))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream()) # submit the backend coroutine
            
            # Stream AI messages
            while True:
                item = event_queue.get()
                if item is None:
                    break

                message_chunk, metadata = item

                if message_chunk == "error":
                    raise metadata

                # Update tool status if applicable
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", None) or getattr(metadata, "tool_name", None) or "tool"
                    
                    # Add tool name to list of used tools
                    if tool_name not in status_holder["used_tools"]:
                        status_holder["used_tools"].append(tool_name)
                        
                    # Update tool status
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🛠️ Running tool: {tool_name}...", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🛠️ Running tool: {tool_name}...", state="running", expanded=True
                        )
                
                # Stream only AI messages to the chat
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        
        ai_message = st.write_stream(ai_only_stream())
          
        # Update tool status only if any tool was actually used
        if status_holder["box"] is not None:
            if status_holder["used_tools"]:
                tools_str = ", ".join(status_holder["used_tools"])
                status_holder["box"].update(label=f"✅ Tool finished - Tools used: {tools_str}", state="complete", expanded=False)
            else:
                status_holder["box"].update(label="✅ No tools used", state="complete", expanded=False)

    # Save assistant response to history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})