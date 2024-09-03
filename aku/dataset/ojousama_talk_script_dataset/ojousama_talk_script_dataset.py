import csv

from tqdm.contrib import tenumerate
import neologdn


with open('data/raw/ojousamatalkscript200.csv', 'r', encoding="utf-8") as f:
    reader = csv.reader(f)

    contents = []
    for i, row in tenumerate(reader):
        if i == 0:
            continue

        user = neologdn.normalize(row[0])
        ojousama = neologdn.normalize(row[1])

        contents.append(f"<ROLE>User</ROLE>\n{user}\n<ROLE>お嬢様</ROLE>\n{ojousama}<EOS>")

with open('data/processed/ojousama_talk_script_dataset.txt', 'w') as f:
    f.writelines(contents)
