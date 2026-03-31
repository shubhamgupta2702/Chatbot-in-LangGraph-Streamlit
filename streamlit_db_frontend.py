import streamlit as st
from langgraph_db_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# -----------------------------
# Utilities
# -----------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    st.session_state['chat_threads'].append(thread_id)
    st.session_state['chat_titles'][thread_id] = "New Chat"


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id}}
    )
    return state.values.get('messages', [])


# -----------------------------
# Session State Initialization
# -----------------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = {}

# Initialize first thread
if not st.session_state['chat_threads']:
    add_thread(st.session_state['thread_id'])


# -----------------------------
# Sidebar UI
# -----------------------------

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads'][::-1]:
    title = st.session_state['chat_titles'].get(thread_id, "Untitled Chat")

    if st.sidebar.button(title, key=f"thread_{thread_id}"):
        st.session_state['thread_id'] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            elif isinstance(msg, AIMessage):
                role = 'assistant'
            else:
                continue

            temp_messages.append({
                'role': role,
                'content': msg.content
            })

        st.session_state['message_history'] = temp_messages

        # Auto-create title if missing
        if thread_id not in st.session_state['chat_titles'] and temp_messages:
            first_user_msg = next(
                (m['content'] for m in temp_messages if m['role'] == 'user'),
                "Untitled Chat"
            )
            st.session_state['chat_titles'][thread_id] = first_user_msg[:40]


# -----------------------------
# Display Chat Messages
# -----------------------------

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


# -----------------------------
# Chat Input
# -----------------------------

user_input = st.chat_input("Type here")

if user_input:

    # Save user message
    st.session_state['message_history'].append({
        'role': "user",
        'content': user_input
    })

    with st.chat_message('user'):
        st.markdown(user_input)

    # Set title if first message
    current_thread = st.session_state['thread_id']
    if st.session_state['chat_titles'].get(current_thread, "New Chat") == "New Chat":
        title = user_input[:40]
        if len(user_input) > 40:
            title += "..."
        st.session_state['chat_titles'][current_thread] = title

    CONFIG = {
        "configurable": {
            "thread_id": current_thread
        }
    }

    # Stream AI response properly
    full_response = ""
    placeholder = st.empty()

    with st.chat_message('assistant'):
      placeholder = st.empty()  # ✅ move inside

    full_response = ""
    for message_chunk, metadata in chatbot.stream(
        {"messages": HumanMessage(content=user_input)},
        config=CONFIG,
        stream_mode='messages'
    ):
        if message_chunk.content:
            full_response += message_chunk.content
            placeholder.markdown(full_response)

    # Save AI response
    st.session_state['message_history'].append({
        'role': "assistant",
        'content': full_response
    })