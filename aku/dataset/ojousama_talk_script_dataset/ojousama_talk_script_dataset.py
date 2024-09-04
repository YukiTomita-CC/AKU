import csv

from tqdm.contrib import tenumerate
import neologdn

from aku.dataset.common.prompt_formatter import format_prompt


def main():
    # Download from https://github.com/matsuvr/OjousamaTalkScriptDataset/blob/main/ojousamatalkscript200.csv
    with open('data/raw/ojousamatalkscript200.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)

        contents = []
        for i, row in tenumerate(reader):
            if i == 0: # Skip header
                continue

            user = neologdn.normalize(row[0])
            ojousama = neologdn.normalize(row[1])

            conversation = [
                {"role": "User", "content": user},
                {"role": "お嬢様", "content": ojousama}
            ]

            contents.append(format_prompt(conversation))

    with open('data/processed/ojousama_talk_script_dataset.txt', 'w', encoding="utf-8") as f:
        f.writelines(contents)


if __name__ == "__main__":
    main()
