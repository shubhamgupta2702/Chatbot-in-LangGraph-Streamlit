import streamlit as st
from langgraph_backend_tools import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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

    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )