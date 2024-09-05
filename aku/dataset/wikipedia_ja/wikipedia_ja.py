# refer: https://github.com/ce-lery/japanese-mistral-300m-recipe/tree/main/pretrain/dataset

from datasets import load_dataset
import neologdn
from textformatting import ssplit
from tqdm import tqdm


def main():
    dataset = load_dataset("wikimedia/wikipedia", data_files="20231101.ja/train-*-of-00015.parquet")

    print("INFO: processing phase1")
    contents = []
    for i in tqdm(range(len(dataset['train']))):
        text = dataset['train'][i]['text']
        if "脚注" in text: # Remove after the footnote section
            contents.append(text[:text.index("脚注")])
        else:
            contents.append(text)

    print("INFO: processing phase2")
    processed_lines = []
    for line in tqdm(contents):
        line = line.replace('\u3000', ' ').replace('\t', ' ').strip()
        line = ' '.join(line.split())

        line = neologdn.normalize(line)
        sentences = ssplit(line)

        processed_lines.extend(sentences)

    with open("data/processed/wikipedia_ja.txt", 'w', encoding='utf-8') as f:
        for sentence in processed_lines:
            f.write(sentence + '\n')


if __name__ == "__main__":
    main()
