import json
import os

# import pyperclip
import streamlit as st


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

emojis = ["😊","🥰","😉","🤗","😭","🤣","😆","🙄","😡","😲","😳","😔","😇","😙","🥳","🤔","🥺","🥱","💓","✨"]
likability_maps = {
    1: "**1. 知らない人、初めて会う人**\n- 初対面の人と話すような口調、敬語\n- 個人的な質問は避け、一般的な話題に留める\n- 距離を保った丁寧な対応",
    3: "**3. 友達**\n- 仲の良い友達や知り合いと話すような口調、くだけた表現\n- 冗談や軽いからかいも可能\n- 個人的な話題も自然に出せる",
    5: "**5. パートナー**\n- 恋人と話すような口調、くだけた表現\n- 愛称や特別な呼び方を使用することも\n- 非常に個人的な話題や感情を共有できる"
    }
mood_maps = {
    1: "**1. 機嫌がとても悪い**\n- イライラした様子、ネガティブな発言、最低限の会話、絵文字や感嘆符は使用しない",
    3: "**3. 機嫌は良くも悪くもない**\n- 通常の会話、絵文字や感嘆符を適度に使用する",
    5: "**5. 機嫌がとても良い**\n- ポジティブな言葉遣い、絵文字や感嘆符を多用する"
}


# ===== Functions =====
def load_user_persona():
    with open("docs/user_persona.json", 'r', encoding='utf-8') as json_file:
        user_personas = json.load(json_file)
    
    return user_personas

def add_emoji(emoji):
    # pyperclip.copy(emoji)
    pass

def load_conversation_filepaths():
    base_dir = "aku/dataset/original/conversations"
    if not os.path.exists(base_dir):
        return []
    
    logs = []
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".json"):
            logs.append(os.path.join(base_dir, file_name))

    return logs

def load_conversation():
    file_path = st.session_state.current_conversation
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    st.session_state.current_dialogs = []
    st.session_state.likability = data["conversations"][1]["attribute"]["likability"]
    st.session_state.mood = data["conversations"][1]["attribute"]["mood"]

    try:
        st.session_state.user_persona = st.session_state.user_personas[data["user_persona"]]
    except KeyError:
        pass

    for conv in data["conversations"]:
        role = conv["role"]
        content = conv["content"]
        st.session_state.current_dialogs.append({"role": role, "content": content})

def save_dialogs():
    conversations = []
    for i in range(len(st.session_state.current_dialogs)):
        if i % 2 == 0:
            conversations.append({
                "role": "user",
                "content": st.session_state[f"dialog_{i}"]
            })
        else:
            conversations.append({
                "role": "assistant",
                "content": st.session_state[f"dialog_{i}"],
                "attribute": {
                    "likability": st.session_state.likability,
                    "mood": st.session_state.mood
                }
            })
    
    data = {"conversations": conversations}

    base_dir = 'aku/dataset/original/edited_conversations'
    os.makedirs(base_dir, exist_ok=True)

    file_num = 0
    while os.path.exists(f"{base_dir}/conv_{file_num}.json"):
        file_num += 1
    file_path = f"{base_dir}/conv_{file_num}.json"

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    
    st.session_state.current_dialogs = []
    st.session_state.likability = 3
    st.session_state.mood = 3
    st.session_state.user_persona = None


# ===== State =====
if 'current_dialogs' not in st.session_state:
    st.session_state.current_dialogs = []

if 'current_conversation' not in st.session_state:
    st.session_state.current_conversation = None

if 'likability' not in st.session_state:
    st.session_state.likability = 3
if 'mood' not in st.session_state:
    st.session_state.mood = 3
if 'user_persona' not in st.session_state:
    st.session_state.user_persona = None

if 'assistant_icon' not in st.session_state:
    icon_path = "aku/dataset/original/assets/assistant_icon.png"
    st.session_state.assistant_icon = icon_path if os.path.exists(icon_path) else None

if 'conversations_files' not in st.session_state:
    st.session_state.conversations_files = load_conversation_filepaths()

if 'user_personas' not in st.session_state:
    st.session_state.user_personas = load_user_persona()


# ===== UI =====
for emoji in emojis:
    st.sidebar.button(emoji, on_click=add_emoji, args=emoji, use_container_width=True)

chat, information = st.columns(2)
with chat:
    with st.container(height=740, border=True):
        for i, dialog in enumerate(st.session_state.current_dialogs):
            avatar = None if dialog["role"] == "user" else st.session_state.assistant_icon
            with st.chat_message(dialog["role"], avatar=avatar):
                st.text_area(
                    "attr",
                    dialog["content"],
                    key=f"dialog_{i}",
                    label_visibility="collapsed"
                )

    st.button(
        "Save",
        on_click=save_dialogs,
        type="primary",
        use_container_width=True
        )

with information:
    st.selectbox(
        "Conversation Files",
        options=st.session_state.conversations_files,
        key="current_conversation",
        on_change=load_conversation,
        label_visibility="collapsed"
    )

    with st.container(height=750, border=True):
        st.write("#### Likability")
        st.write(likability_maps[st.session_state.likability])
        st.write("#### Mood")
        st.write(mood_maps[st.session_state.mood])
        st.write("#### User Persona")
        st.write(f"```json\n{json.dumps(st.session_state.user_persona, ensure_ascii=False, indent=2)}\n```")
