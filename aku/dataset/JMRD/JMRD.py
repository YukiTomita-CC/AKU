import json
import os

from tqdm import tqdm
import neologdn

from aku.dataset.common.alphabet_iterator import AlphabetIterator
from aku.dataset.common.output_check import has_lines_no_start_with_role
from aku.dataset.common.prompt_formatter import format_prompt


def main():
    # Download from https://github.com/ku-nlp/JMRD
    target_directory = "data/raw/JMRD/data"

    json_files = []
    for dirpath, dirnames, filenames in os.walk(target_directory):
        for filename in filenames:
            if filename.endswith('.json'):
                json_files.append(os.path.join(dirpath, filename))

    alphabet_iterator = AlphabetIterator()
    contents = []
    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for dialog in tqdm(data):
                conversation = []
                alphabet = alphabet_iterator.next()
                for d in dialog["dialog"]:
                    conversation.append({
                        "role": f"推薦者{alphabet}" if d["speaker"] == "recommender" else f"質問者{alphabet}",
                        "content": neologdn.normalize(d["text"].replace("\n", ""))
                    })
        
                # Skip 2 turn conversation because it is like "Hello" and "Nice to meet you"
                # Remove the last turn because it is like "Thank you" and "Thank you too"
                contents.append(format_prompt(conversation[4:-2]))

    with open('data/processed/JMRD.txt', 'w', encoding="utf-8") as f:
        f.writelines(contents)

    if has_lines_no_start_with_role('data/processed/JMRD.txt'):
        print("Warning: Some lines do not start with <ROLE>.")


if __name__ == "__main__":
    main()
