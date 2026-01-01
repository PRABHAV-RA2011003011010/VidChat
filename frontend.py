import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from urllib.parse import urlparse, parse_qs
import uuid
from datetime import datetime

from backend import chatbot
from backend import fetch_video_transcript
from backend import fetch_yt_title
# **************************************** Water Marker ********************************
st.markdown(
    """
    <style>
    .watermark {
        position: fixed;
        bottom: 15px;
        right: 20px;
        opacity: 0.15;
        font-size: 18px;
        pointer-events: none;
        z-index: 9999;
    }
    </style>

    <div class="watermark">
        VidChat • Built by Prabhav
    </div>
    """,
    unsafe_allow_html=True
)

# **************************************** Utility Functions ********************************

def YTVideo_ID_generator(video_url):
    parsed = urlparse(video_url)
    video_id = parse_qs(parsed.query).get("v", [None])[0]
    return(video_id)  

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state.pop("video_url", None)
    st.session_state['message_history'] = []
    if "video_titles" in st.session_state:
        st.session_state["video_titles"][str(thread_id)] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        
def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

def handle_video_submit():
    video_url = st.session_state["video_url"]
    if not video_url:
        return

    video_id = YTVideo_ID_generator(video_url)
    if not video_id:
        st.error("Invalid YouTube URL")
        return

    fetch_video_transcript(video_id, str(st.session_state["thread_id"]))

    title = fetch_yt_title(video_id)
    chat_id = str(st.session_state["thread_id"])

    if title:
        if chat_id not in st.session_state["video_titles"]:
            st.session_state["video_titles"][chat_id] = []

        if title not in st.session_state["video_titles"][chat_id]:
            st.session_state["video_titles"][chat_id].append(title)

    st.session_state.pop("video_url", None)
    st.success("Video loaded successfully")    

@st.dialog("📹 Load YouTube Video")
def video_upload_popup():
    st.text_area(
        "Paste YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        height=120,
        key="video_url"
    )

    if st.button("Load Video", use_container_width=True):
        handle_video_submit()
        st.rerun()  # ✅ closes popup
        

# **************************************** Session Setup **************************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
    
if "video_url" not in st.session_state:
    st.session_state["video_url"] = ""

if "video_titles" not in st.session_state:
    st.session_state["video_titles"] = {}
    
add_thread(st.session_state['thread_id'])

# **************************************** Frontend Code and Logic ******************************
                   
st.sidebar.title('VidChat Options')

st.sidebar.button("New Chat", on_click=reset_chat)

if st.sidebar.button("📹 Paste Video Link"):
    video_upload_popup()

if st.sidebar.button("📚 Context"):
    st.session_state["show_context"] = not st.session_state.get("show_context", False)

# Show loaded videos if context is toggled
if st.session_state.get("show_context", False):
    chat_id = str(st.session_state["thread_id"])
    titles = st.session_state["video_titles"].get(chat_id, [])
    
    st.sidebar.markdown("### Loaded Videos")
    if titles:
        for t in titles:
            st.sidebar.caption(f"• {t}")
    else:
        st.sidebar.caption("No videos loaded")
    
st.sidebar.header('My Conversations')

st.markdown("""
<style>
/* Chat buttons */
.chat-btn {
    width: 100%;
    padding: 10px;
    margin-bottom: 6px;
    border-radius: 14px;
    border: none;
    font-weight: 500;
    cursor: pointer;
    text-align: center;
    background-color: #ffffff;
}

/* Hover */
.chat-btn:hover {
    background-color: #f2f2f2;
}

/* Active chat */
.chat-btn.active {
    background-color: #ff9800;
    color: white;
    font-weight: 700;
    cursor: default;
}
</style>
""", unsafe_allow_html=True)



ChatID = 1
for thread_id in st.session_state["chat_threads"]:
    is_current = (thread_id == st.session_state["thread_id"])

    if is_current:
        st.sidebar.markdown(
            f"""
            <button class="chat-btn active">
                Chat - {ChatID}
            </button>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(
            f"Chat - {ChatID}",
            key=f"chat_{thread_id}",
            use_container_width=True
        ):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_messages.append({"role": role, "content": msg.content})

            st.session_state["message_history"] = temp_messages
            st.rerun()  # ✅ IMPORTANT

    ChatID += 1

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'User_Input'}
#{'role': 'assistant', 'content': 'LLM_Response'}

user_input = st.chat_input("Type Your Query")



if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    
    #ai_message = response['messages'][-1].content
    #st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    with st.chat_message('assistant'):
        
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {
                    'chat_id': str(st.session_state['thread_id']),
                    'messages': [HumanMessage(content=user_input)]
                },
                config= CONFIG,
                stream_mode= 'messages'
            )
        )
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
 
 