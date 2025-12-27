import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from urllib.parse import urlparse, parse_qs
import uuid

from backend import chatbot
from backend import fetch_video_transcript

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
    st.session_state['video_url'] = ""
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        
def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# **************************************** Session Setup **************************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []


    
add_thread(st.session_state['thread_id'])

# **************************************** Frontend Code and Logic ******************************

top_container = st.container()

with top_container:
    left, center, right = st.columns([1, 2, 1])

    with center:
        st.subheader("🧠 Ask something")

        video_url = st.text_area(
            "Give your YT_Video_Link",
            placeholder="Type your message here...",
            height=120
        )

        send_clicked = st.button("Send", use_container_width=True)

        if send_clicked and video_url:
            video_id = YTVideo_ID_generator(video_url)
            if video_id:
                fetch_video_transcript(video_id, str(st.session_state['thread_id']))
                st.success(f"Video {video_id} Loaded")
                
            else:
                st.error("Invalid YouTube URL")
                
        
st.sidebar.title('VidChat Chats')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'User_Input'}
#{'role': 'assistant', 'content': 'LLM_Response'}


user_input = st.chat_input('Type Your Query')

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
        