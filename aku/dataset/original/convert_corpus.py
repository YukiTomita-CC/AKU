import json
import os

from tqdm import tqdm

from aku.dataset.common.output_check import has_lines_no_start_with_role
from aku.dataset.common.prompt_formatter import format_prompt


def main():
    target_directory = "aku/dataset/original/conversations"

    json_files = []
    for dirpath, dirnames, filenames in os.walk(target_directory):
        for filename in filenames:
            if filename.endswith('.json'):
                json_files.append(os.path.join(dirpath, filename))

    contents = []
    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            conversation = []

            data = json.load(f)
            utterances = data["conversation"]
            for utterance in utterances:
                conversation.append({
                    "role": "User" if utterance["role"] == "user" else "Aku",
                    "content": utterance["content"].replace("？", "?").replace("！", "!")
                })
        
        contents.append(format_prompt(conversation))

    with open('data/processed/original.txt', 'w', encoding="utf-8") as f:
        f.writelines(contents)

    if has_lines_no_start_with_role('data/processed/original.txt'):
        print("Warning: Some lines do not start with <ROLE>.")


if __name__ == "__main__":
    main()
