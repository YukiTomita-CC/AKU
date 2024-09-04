import json
import os
import re

from tqdm import tqdm
import neologdn

from aku.dataset.common.alphabet_iterator import AlphabetIterator
from aku.dataset.common.output_check import has_lines_no_start_with_role
from aku.dataset.common.prompt_formatter import format_prompt


def _contains_bracketed_text(text) -> bool:
    """Check if the text contains bracketed text.\\
    Like this: Hi \<AA\>. how are you?

    Args:
        text str: The text to check.

    Returns:
        bool: True if the text contains bracketed text.
    """
    pattern = r'<[^>]+>'
    if re.search(pattern, text):
        return True
    return False

def _contains_asterisk(text) -> bool:
    """Check if the text contains asterisk.\\
    Like this: My name is ＊＊.

    Args:
        text str: The text to check.

    Returns:
        bool: True if the text contains asterisk.
    """
    if "＊＊" in text:
        return True
    return False

def main():
    # Download from https://github.com/nu-dialogue/real-persona-chat
    target_directory = "data/raw/real-persona-chat/real_persona_chat/dialogues"

    json_files = []
    for dirpath, dirnames, filenames in os.walk(target_directory):
        for filename in filenames:
            if filename.endswith('.json'):
                json_files.append(os.path.join(dirpath, filename))

    alphabet_iterator = AlphabetIterator()
    contents = []
    for json_file in tqdm(json_files):
        is_find_excluded_word = False
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            interlocutors = data["interlocutors"]

            alphabet = alphabet_iterator.next()
            conversation = []
            for utterance in data["utterances"]:
                # refer: https://github.com/nu-dialogue/real-persona-chat/tree/main?tab=readme-ov-file#%E5%AF%BE%E8%A9%B1%E3%83%87%E3%83%BC%E3%82%BF
                if _contains_bracketed_text(utterance["text"]) or _contains_asterisk(utterance["text"]):
                    is_find_excluded_word = True
                    break

                conversation.append({
                    "role": f"User" if utterance["interlocutor_id"] == interlocutors[0] else f"{alphabet}さん",
                    "content": neologdn.normalize(utterance["text"].replace("\n", ""))
                })

            if not is_find_excluded_word:
                # Skip 2 turn conversation because it is like "Hello" and "Nice to meet you"
                contents.append(format_prompt(conversation[2:]))

    with open('data/processed/real-persona-chat.txt', 'w', encoding="utf-8") as f:
        f.writelines(contents)

    if has_lines_no_start_with_role('data/processed/real-persona-chat.txt'):
        print("Warning: Some lines do not start with <ROLE>.")


if __name__ == "__main__":
    main()
