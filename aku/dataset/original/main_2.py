import json
import os

import streamlit as st

from aku_model import AkuModel


st.set_page_config(layout="wide")

image_size = 3.5 # default=2
image_message_gap = 0.5 # default=0.5
CHAT_MESSAGE_STYLE = f"""<style>
img.eeusbqq0 {{
    width: {image_size}rem !important;
    height: {image_size}rem !important;
}}

.stChatMessage.eeusbqq4 {{
    gap: {image_message_gap}rem !important;
}}
</style>
"""
st.markdown(CHAT_MESSAGE_STYLE, unsafe_allow_html=True)

emojis = ["😊", "🥰", "😉", "🤗", "😭", "🤣", "😲", "😳", "🥺", "💓", "✨", "😆", "🙄", "😡", "😔"]


def add_emoji(emoji):
    st.session_state.input_message += emoji

def send_message():
    role = "user" if len(st.session_state.current_dialogs) % 2 == 0 else "assistant"
    
    st.session_state.current_dialogs.append({"role": role, "content": st.session_state.input_message})
    st.session_state.input_message = ""

    if st.session_state.model is not None and len(st.session_state.current_dialogs) % 2 == 1:
        response = st.session_state.model.batch_inference(
            {
                "context": st.session_state.context,
                "dialogs": st.session_state.current_dialogs,
                "likability": st.session_state.likability,
                "mood": st.session_state.mood
            }
        )
        st.session_state.model_responses = response

def save_dialogs():
    conversations = []
    for i in range(len(st.session_state.current_dialogs)):
        if i % 2 == 0:
            conversations.append({
                "role": "user",
                "content": st.session_state.current_dialogs[i]['content']
            })
        else:
            conversations.append({
                "role": "assistant",
                "content": st.session_state.current_dialogs[i]['content'],
                "attribute": {
                    "likability": st.session_state.likability,
                    "mood": st.session_state.mood
                }
            })
    
    data = {
        "context": st.session_state.context,
        "conversations": conversations
    }

    base_dir = 'aku/dataset/original/conversations'
    os.makedirs(base_dir, exist_ok=True)

    file_num = 0
    while os.path.exists(f"{base_dir}/conv_{file_num}.json"):
        file_num += 1
    file_path = f"{base_dir}/conv_{file_num}.json"

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    
    st.session_state.current_dialogs = []
    st.session_state.context = ""
    st.session_state.likability = 3
    st.session_state.mood = 3

def choose_gen(gen):
    response = st.session_state.model_responses[gen - 1]
    st.session_state.current_dialogs.append({"role": "assistant", "content": response})

def format_likeability(likability):
    format_map = {
        1: "知らないor初めて会う人",
        2: "数回会ったことある人",
        3: "よく会う人",
        4: "友達",
        5: "パートナー"
    }

    return "User: " + format_map[likability]

def format_mood(mood):
    format_map = {
        1: "最悪",
        2: "悪い",
        3: "普通",
        4: "良い",
        5: "最高"
    }

    return "Akuの機嫌: " + format_map[mood]

def load_model():
    model_name = st.session_state.selected_model
    if model_name is None:
        st.session_state.model = None
        return
    
    base_dir = "models"
    model_dir = os.path.join(base_dir, model_name)

    if not os.path.exists(model_dir):
        st.error(f"Model {model_name} does not exist.")
        return
    
    st.session_state.model = AkuModel(model_dir)

if 'current_dialogs' not in st.session_state:
    st.session_state.current_dialogs = []

if 'input_message' not in st.session_state:
    st.session_state.input_message = ""

if 'context' not in st.session_state:
    st.session_state.context = ""
if 'likability' not in st.session_state:
    st.session_state.likability = 3
if 'mood' not in st.session_state:
    st.session_state.mood = 3

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'model' not in st.session_state:
    st.session_state.model = None

if 'model_responses' not in st.session_state:
    st.session_state.model_responses = []


for emoji in emojis:
    st.sidebar.button(emoji, on_click=add_emoji, args=emoji, use_container_width=True)

chat, generate = st.columns(2)
with chat:
    st.text_input(
        "context",
        key="context",
        placeholder="context",
        label_visibility="collapsed"
        )
    
    likability, mood, save_button = st.columns([2, 2, 1])
    with likability:
        st.selectbox(
            "Likability",
            [1, 2, 3, 4, 5],
            format_func=format_likeability,
            key="likability",
            label_visibility="collapsed"
            )
    with mood:
        st.selectbox(
            "Mood",
            [1, 2, 3, 4, 5],
            format_func=format_mood,
            key="mood",
            label_visibility="collapsed"
            )
    with save_button:
        st.button(
            "**!! Save !!**",
            on_click=save_dialogs,
            type="primary",
            use_container_width=True
            )

    with st.container(height=630, border=True):
        for i, dialog in enumerate(st.session_state.current_dialogs):
            avatar = None if dialog["role"] == "user" else "aku/dataset/original/assets/assistant_icon.png"
            with st.chat_message(dialog["role"], avatar=avatar):
                st.write(dialog["content"])
        st.write(f"Conversation Turns: {len(st.session_state.current_dialogs) // 2}")
    
    text_input, send_button = st.columns([9, 1])
    with text_input:
        prompt = st.text_input(
            "Message",
            key='input_message',
            label_visibility="collapsed")
    with send_button:
        st.button(
            "Send",
            on_click=send_message,
            type="primary",
            use_container_width=True
            )

with generate:
    st.selectbox(
        "Model",
        [None, "inc_batch_3epoch"],
        index=0,
        key="selected_model",
        on_change=load_model,
        label_visibility="collapsed"
        )
    
    with st.container(height=685, border=True):
        for i, response in enumerate(st.session_state.model_responses):
            st.write(f"#### {i + 1}st generation")
            with st.chat_message("assistant"):
                st.write(response)
    
    gen1, gen2, gen3, gen4, gen5 = st.columns(5)
    with gen1:
        st.button("Gen1", on_click=choose_gen, kwargs={"gen": 1}, use_container_width=True)
    with gen2:
        st.button("Gen2", on_click=choose_gen, kwargs={"gen": 2}, use_container_width=True)
    with gen3:
        st.button("Gen3", on_click=choose_gen, kwargs={"gen": 3}, use_container_width=True)
    with gen4:
        st.button("Gen4", on_click=choose_gen, kwargs={"gen": 4}, use_container_width=True)
    with gen5:
        st.button("Gen5", on_click=choose_gen, kwargs={"gen": 5}, use_container_width=True)
