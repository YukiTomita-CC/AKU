import json
import os
import streamlit as st

from aku_chat_model import AkuChatModel


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

# ===== Functions =====
def send_message():
    role = "user" if len(st.session_state.current_dialogs) % 2 == 0 else "assistant"
    
    st.session_state.current_dialogs.append({"role": role, "content": st.session_state.input_message})
    st.session_state.input_message = ""

    outputs = st.session_state.inference_model.generate_responses(st.session_state.current_dialogs)
    st.session_state.model_outputs = outputs

def save_record():
    if st.session_state.best_responses_num == st.session_state.worst_responses_num:
        return
    
    record = {
        "input": st.session_state.current_dialogs,
        "chosen": st.session_state.model_outputs[st.session_state.best_responses_num],
        "rejected": st.session_state.model_outputs[st.session_state.worst_responses_num]
    }

    base_dir = 'aku/fine_tuning/dpo_data'
    os.makedirs(base_dir, exist_ok=True)

    file_num = 0
    while os.path.exists(f"{base_dir}/record_{file_num}.json"):
        file_num += 1

    file_path = f"{base_dir}/record_{file_num}.json"

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    
    st.session_state.best_responses_num = 0
    st.session_state.worst_responses_num = 0


# ===== State =====
if "model_outputs" not in st.session_state:
    st.session_state.model_outputs = []

if "current_dialogs" not in st.session_state:
    st.session_state.current_dialogs = []

if "inference_model" not in st.session_state:
    st.session_state.inference_model = AkuChatModel()

if "best_responses_num" not in st.session_state:
    st.session_state.best_responses_num = 0

if "worst_responses_num" not in st.session_state:
    st.session_state.worst_responses_num = 0

if 'assistant_icon' not in st.session_state:
    icon_path = "aku/dataset/original/assets/assistant_icon.png"
    st.session_state.assistant_icon = icon_path if os.path.exists(icon_path) else None


# ===== UI =====
select, outputs = st.columns(2)

with select:
    with st.container(height=250, border=True):
        st.radio(
            "Select Best Responses",
            range(10),
            key="best_responses_num",
            horizontal=True,
        )
        st.radio(
            "Select Worst Responses",
            range(10),
            key="worst_responses_num",
            horizontal=True,
        )
        st.button(
            "Save",
            on_click=save_record,
            type="primary",
        )

    with st.container(height=450, border=True):
        for i, dialog in enumerate(st.session_state.current_dialogs):
            avatar = None if dialog["role"] == "user" else st.session_state.assistant_icon
            with st.chat_message(dialog["role"], avatar=avatar):
                st.write(dialog["content"])
    
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

with outputs:
    with st.container(height=780, border=True):
        for i, response in enumerate(st.session_state.model_outputs):
            st.write(f"#### {i}")
            with st.chat_message("assistant", avatar=st.session_state.assistant_icon):
                st.write(response)
