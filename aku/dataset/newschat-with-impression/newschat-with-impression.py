import json

from tqdm import tqdm
import neologdn

from aku.dataset.common.alphabet_iterator import AlphabetIterator
from aku.dataset.common.output_check import has_lines_no_start_with_role
from aku.dataset.common.prompt_formatter import format_prompt


def main():
    # Download from https://github.com/fukanarita/newschat-with-impression
    jsonl_files = ["data/raw/newschat-with-impression/V2.jsonl", "data/raw/newschat-with-impression/V3.jsonl"]

    alphabet_iterator = AlphabetIterator()
    contents = []
    for jsonl_file in tqdm(jsonl_files):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                conversation = []
                alphabet = alphabet_iterator.next()

                dialog = json.loads(line)

                for d in dialog["dialog"]:
                    conversation.append({
                        "role": f"{alphabet}さん" if d["speaker"] == "S" else "User",
                        "content": neologdn.normalize(d["utterance"].replace("\n", ""))
                    })

                contents.append(format_prompt(conversation))

    with open('data/processed/newschat-with-impression.txt', 'w', encoding="utf-8") as f:
        f.writelines(contents)

    if has_lines_no_start_with_role('data/processed/newschat-with-impression.txt'):
        print("Warning: Some lines do not start with <ROLE>.")


if __name__ == "__main__":
    main()
