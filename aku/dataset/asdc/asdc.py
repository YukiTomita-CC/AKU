import json
import os

from tqdm import tqdm
import neologdn

from aku.dataset.common.output_check import has_lines_no_start_with_role
from aku.dataset.common.prompt_formatter import format_prompt


def _convert_name(name):
    if "operator" in name:
        return name.replace("operator_", "オペレーター")
    else:
        return name.replace("customer_", "カスタマー")

def main():
    # Download from https://github.com/megagonlabs/asdc
    target_directory = "data/raw/asdc/data/main/dialog/json"

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
            utterances = data["utterances"]
            for utterance in utterances:
                if "text_fixed" in utterance:
                    print("INFO: Using text_fixed field.")
                    conversation.append({
                        "role": _convert_name(utterance["name"]),
                        "content": neologdn.normalize(utterance["text_fixed"].replace("\n", ""))
                    })
                else:
                    conversation.append({
                        "role": _convert_name(utterance["name"]),
                        "content": neologdn.normalize(utterance["text"].replace("\n", ""))
                    })
        
        # Remove the first utterance and the last 2 because they are very similar.
        contents.append(format_prompt(conversation[1:-2]))

    with open('data/processed/asdc.txt', 'w', encoding="utf-8") as f:
        f.writelines(contents)

    if has_lines_no_start_with_role('data/processed/asdc.txt'):
        print("Warning: Some lines do not start with <ROLE>.")


if __name__ == "__main__":
    main()
