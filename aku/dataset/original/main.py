import json
import os

import pyperclip
import streamlit as st


image_size = 5 # default=2
image_message_gap = 1.5 # default=0.5
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


def get_context():
    contexts = [
        "",
        "最近見た映画やドラマの感想をAに語る",
        "資格取得を目指して勉強している人が、その進捗や悩みをAに話す",
        "子供が新しいゲームやおもちゃについてAに説明する",
        "季節の変わり目に、体調管理や季節の行事についてAと話す",
        "新しい料理に挑戦した人が、その成功をAに報告する",
        "最近観光地を訪れた人が、観光地の様子や感じたことをAに説明する",
        "親友との関係について悩んでいる人が、Aにアドバイスを求める",
        "学生が期末試験やレポートの準備についてAに相談する",
        "音楽フェスに参加した人が、その感想や出会ったアーティストについてAに語る",
        "新しいテクノロジーやガジェットを手に入れた人が、それを試してみた感想をAに伝える",
        "スポーツチームに加入した人が、練習や試合の様子をAに話す",
        "新しいレストランに行った人が、料理の味や雰囲気をAに伝える",
        "大学のサークル活動の報告や進捗をAに話す",
        "家族との関係に悩んでいる人が、Aにその思いを打ち明ける",
        "健康診断の結果や健康に関する話題について、Aに相談する",
        "近所のカフェでアルバイトを始めた人が、初日の体験をAに話す",
        "電車の中で面白い出来事に出くわした人が、その一部始終をAに伝える",
        "飼っているペットが芸を覚えたことをAに自慢げに報告する",
        "いつも使っている家電が壊れてしまい、困っていることをAに相談する",
        "インターネットで見つけた面白いニュースや情報をAに共有する",
        "最近ハマっているスマホゲームの攻略法についてAと議論する",
        "夢を見た人が、その内容をAに説明し、夢判断をしてもらう",
        "コンビニで新発売のお菓子を見つけたので、Aにも勧める",
        "好きなアイドルのコンサートチケットが当たったことをAに興奮気味に報告する",
        "苦手な家事をAに手伝ってもらえないか相談する",
        "最近行った美容院で新しいヘアスタイルにしたことをAに見てもらう",
        "体調が悪く、病院に行くべきかAに相談する",
        "近所で行われる祭りやイベントにAを誘う",
        "ネットショッピングで失敗したエピソードをAに面白おかしく語る",
        "料理のレシピをAに教えてもらう",
        "自分の好きな音楽やアーティストをAに紹介する",
        "将来の夢や目標についてAと語り合う",
        "ストレスが溜まっていることをAに打ち明け、話を聞いてもらう",
        "最近読んだ漫画や小説のストーリーをAに熱く語る",
        "突然宝くじが当たった人が、その後の計画についてAに相談する",
        "宗教や信仰についてAと語り合う",
        "死後の世界や輪廻転生についてAと議論する",
        "人生の意味や幸福についてAと哲学的な議論をする",
        "社会問題や環境問題についてAと意見交換をする"
    ]

    context_id = len([name for name in os.listdir('aku/dataset/original/conversations')])

    return contexts[context_id]


def save_conversation(conversation, context):
    data = {
        "context": context,
        "conversation": conversation
    }

    base_dir = 'aku/dataset/original/conversations'
    os.makedirs(base_dir, exist_ok=True)

    if 'file_path' not in st.session_state:
        file_num = 0
        while os.path.exists(f"{base_dir}/conv_{file_num}.json"):
            file_num += 1
        st.session_state['file_path'] = f"{base_dir}/conv_{file_num}.json"

    file_path = st.session_state['file_path']
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    
    st.session_state['messages'] = []

def add_emoji(emoji):
    pyperclip.copy(emoji)


# Session state initialization
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'context' not in st.session_state:
    st.session_state['context'] = get_context()

# Main app
st.sidebar.button("Save conversation", on_click=save_conversation, args=(st.session_state['messages'], st.session_state['context']))

st.sidebar.markdown(f"#### {st.session_state['context']}")
st.sidebar.markdown("#### ")
st.sidebar.markdown(f"Turns Count: {len(st.session_state['messages']) // 2}")

st.sidebar.markdown("---")
st.sidebar.button("😊", on_click=add_emoji, args="😊", use_container_width=True)
st.sidebar.button("🥰", on_click=add_emoji, args="🥰", use_container_width=True)
st.sidebar.button("😉", on_click=add_emoji, args="😉", use_container_width=True)
st.sidebar.button("🤗", on_click=add_emoji, args="🤗", use_container_width=True)
st.sidebar.button("😭", on_click=add_emoji, args="😭", use_container_width=True)
st.sidebar.button("🤣", on_click=add_emoji, args="🤣", use_container_width=True)
st.sidebar.button("😲", on_click=add_emoji, args="😲", use_container_width=True)
st.sidebar.button("😳", on_click=add_emoji, args="😳", use_container_width=True)
st.sidebar.button("🥺", on_click=add_emoji, args="🥺", use_container_width=True)
st.sidebar.button("💓", on_click=add_emoji, args="💓", use_container_width=True)
st.sidebar.button("✨", on_click=add_emoji, args="✨", use_container_width=True)
st.sidebar.button("😆", on_click=add_emoji, args="😆", use_container_width=True)
st.sidebar.button("🙄", on_click=add_emoji, args="🙄", use_container_width=True)
st.sidebar.button("😡", on_click=add_emoji, args="😡", use_container_width=True)
st.sidebar.button("😔", on_click=add_emoji, args="😔", use_container_width=True)

for i, message in enumerate(st.session_state['messages']):
    if i % 2 == 0:
        avatar = None
    else:
        if os.path.exists("aku/dataset/original/persona/assistant_icon.png"):
            avatar = "aku/dataset/original/persona/assistant_icon.png"
        else:
            avatar = None

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Message..."):
    if len(st.session_state['messages']) % 2 == 0:
        st.session_state['messages'].append({"role": "user", "content": prompt})
    else:
        st.session_state['messages'].append({"role": "assistant", "content": prompt})
    
    st.rerun()
