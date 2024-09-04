from datasets import load_dataset
from tqdm import tqdm
import neologdn

from aku.dataset.common.alphabet_iterator import AlphabetIterator
from aku.dataset.common.output_check import has_lines_no_start_with_role
from aku.dataset.common.prompt_formatter import format_prompt


def main():
    dataset = load_dataset("nu-dialogue/jmultiwoz", trust_remote_code=True)

    alphabet_iterator = AlphabetIterator()
    contents = []
    for purpose in ["train", "validation", "test"]:
        for n in tqdm(range(len(dataset[purpose]))):
            alphabet = alphabet_iterator.next()

            conversation = []
            for i in range(len(dataset["train"][n]["turns"]["speaker"])):
                conversation.append({
                    "role": "User" if dataset["train"][n]["turns"]["speaker"][i] == "USER" else f"{alphabet}さん",
                    "content": neologdn.normalize(dataset["train"][n]["turns"]["utterance"][i].replace("\n", ""))
                })

            contents.append(format_prompt(conversation))

    with open('data/processed/jmultiwoz.txt', 'w', encoding="utf-8") as f:
        f.writelines(contents)

    if has_lines_no_start_with_role('data/processed/jmultiwoz.txt'):
        print("Warning: Some lines do not start with <ROLE>.")


if __name__ == "__main__":
    main()
