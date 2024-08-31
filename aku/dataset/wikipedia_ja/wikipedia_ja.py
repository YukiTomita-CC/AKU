# refer:https://github.com/ce-lery/japanese-mistral-300m-recipe/tree/main/pretrain/dataset

from datasets import load_dataset
import neologdn
from textformatting import ssplit


def create_raw_data(file_path):
    dataset = load_dataset("wikimedia/wikipedia", data_files="20231101.ja/train-*-of-00015.parquet")

    contents = []
    for i in range(len(dataset['train'])):
        text = dataset['train'][i]['text']
        if "脚注" in text: # 脚注セクション以降は削除
            contents.append(text[:text.index("脚注")])

    with open(file_path, 'w') as f:
        for item in contents:
            f.write(item + '\n')

def process_text(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        line = line.replace('\u3000', ' ').replace('\t', ' ').strip()
        line = ' '.join(line.split())

        if line and len(line) >= 20:
            line = neologdn.normalize(line)
            sentences = ssplit(line)
            processed_lines.extend(sentences)

    with open(file_path, 'w') as f:
        for sentence in processed_lines:
            f.write(sentence + '\n')


if __name__ == "__main__":
    wiki_file = "wikipedia_ja.txt"

    create_raw_data(wiki_file)

    process_text(wiki_file)
