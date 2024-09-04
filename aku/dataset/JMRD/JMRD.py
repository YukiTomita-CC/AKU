import json
import os

from tqdm import tqdm
import neologdn

from aku.dataset.common.output_check import has_lines_no_start_with_role
from aku.dataset.common.prompt_formatter import format_prompt


class AlphabetIterator:
    def __init__(self):
        self.current = ord('A') # ASCII code of 'A'
    
    def next(self):
        letter = chr(self.current)
        self.current += 1
        
        if self.current > ord('Z'):
            self.current = ord('A')
        
        return letter

def main():
    # Download from https://github.com/megagonlabs/asdc
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
            conversation = []

            data = json.load(f)
            for dialogs in tqdm(data):
                alphabet = alphabet_iterator.next()
                for d in dialogs["dialog"]:
                    conversation.append({
                        "role": f"推薦者{alphabet}" if d["speaker"] == "recommender" else f"質問者{alphabet}",
                        "content": neologdn.normalize(d["text"].replace("\n", ""))
                    })
        
                contents.append(format_prompt(conversation))

    with open('data/processed/JMRD.txt', 'w', encoding="utf-8") as f:
        f.writelines(contents)

    if has_lines_no_start_with_role('data/processed/JMRD.txt'):
        print("Warning: Some lines do not start with <ROLE>.")


if __name__ == "__main__":
    main()
