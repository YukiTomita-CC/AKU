from datasets import load_dataset
import neologdn
from textformatting import ssplit
from tqdm import tqdm


def main():
    data = load_dataset("data/raw/aozorabunko-clean")

    def filter_method(example):
        return example["meta"]["文字遣い種別"] == "新字新仮名"
    
    train_data = data["train"].filter(filter_method)

    contents = []
    for i in tqdm(range(len(train_data))):
        line = train_data[i]["text"]
        line = line.replace('\u3000', ' ').replace('\t', ' ').strip()
        line = ' '.join(line.split())

        line = neologdn.normalize(line)
        sentences = ssplit(line)

        contents.extend(sentences)
    
    with open('data/processed/aozorabunko-clean.txt', 'w', encoding="utf-8") as f:
        for sentence in contents:
            f.write(sentence + '\n')


if __name__ == "__main__":
    main()
