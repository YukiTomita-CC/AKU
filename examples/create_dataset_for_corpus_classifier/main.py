import json
import os
import random

import streamlit as st


THEME = "natural"
FILE_PATH = ""
DATA_NUM = 500

# ===== Function =====
def label_line(label):
    st.session_state.annotations.append({
        "line_index": st.session_state.current_line_num,
        "text": st.session_state.data[st.session_state.current_line_num],
        "label": label
    })

    st.session_state.current_line_num += 1

def save_annotations():
    directory = f'data/{THEME}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_list = os.listdir(directory)
    annotation_files = [f for f in file_list if f.endswith(".jsonl")]
    index = len(annotation_files) + 1

    file_path = f'{directory}/annotations_{index}.jsonl'

    with open(file_path, 'w', encoding='utf-8') as f:
        for annotation in st.session_state.annotations:
            f.write(json.dumps(annotation, ensure_ascii=False) + '\n')

    st.success(f"Annotations saved to {file_path}")


# ===== Initialize State =====
if 'data' not in st.session_state:
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if json.loads(line)['is_rejected'] == "0"]
    
    random_samples = random.sample(records, min(DATA_NUM, len(records)))

    text_list = [sample['text'] for sample in random_samples]

    st.session_state.data = text_list

if 'current_line_num' not in st.session_state:
    st.session_state.current_line_num = 0

if 'annotations' not in st.session_state:
    st.session_state.annotations = []


# ===== UI =====
if st.session_state.current_line_num < DATA_NUM:
    st.write(f"{st.session_state.current_line_num + 1} / {DATA_NUM}:")

    st.text_area("text", st.session_state.data[st.session_state.current_line_num], 500)

    col1, col2 = st.columns(2)
    with col1:
        st.button(THEME, on_click=label_line, args=(THEME,), use_container_width=True)
    with col2:
        st.button(f"un{THEME}", on_click=label_line, args=(f"un{THEME}",), use_container_width=True)

else:
    st.write("Finish!!!")

    if st.button("save", type="primary", use_container_width=True):
        save_annotations()
