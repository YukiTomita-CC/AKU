import json
import os

from tqdm import tqdm
import neologdn

from aku.dataset.common.output_check import has_lines_no_start_with_role
from aku.dataset.common.prompt_formatter import format_prompt


def main():
    # Download from https://github.com/matsuvr/characterconversationdataset
    target_directory = "data/raw/characterconversationdataset"

    jsonl_files = []
    for dirpath, dirnames, filenames in os.walk(target_directory):
        for filename in filenames:
            if filename.endswith('.jsonl'):
                jsonl_files.append(os.path.join(dirpath, filename))

    contents = []
    for jsonl_file in tqdm(jsonl_files):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            conversation = []
            for line in f:
                data = json.loads(line)

                conversation.append({
                    "role": data["name"],
                    "content": neologdn.normalize(data["text"].replace("\n", ""))
                })

        contents.append(format_prompt(conversation))

    with open('data/processed/characterconversationdataset.txt', 'w', encoding="utf-8") as f:
        f.writelines(contents)

    if has_lines_no_start_with_role('data/processed/characterconversationdataset.txt'):
        print("Warning: Some lines do not start with <ROLE>.")


if __name__ == "__main__":
    main()
