import json
import os

from datasets import Dataset
from huggingface_hub import login


def convert_role(role):
    if role == "user":
        return "User"
    elif role == "assistant":
        return "Aku"
    else:
        raise ValueError("Invalid role.")

def convert_to_prompt(input_data):
    prompt = ""
    for turn in input_data:
        if turn['role'] == "user":
            prompt += f"<ROLE>{convert_role(turn['role'])}</ROLE>\n{turn['content']}\n"
        else:
            prompt += f"<ROLE>{convert_role(turn['role'])}</ROLE>\n{turn['content']}<EOS>\n"
    
    return prompt + "<ROLE>Aku</ROLE>\n"

def gen(max_datafile_num: int):
    for n in range(max_datafile_num):
        file_path = f"aku/fine_tuning/dpo_data/record_{n}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        yield {
            "prompt": convert_to_prompt(data["input"]),
            "chosen": data["chosen"],
            "rejected": data["rejected"]
        }


if __name__ == "__main__":
    max_datafile_num = 0
    for n in range(2000):
        if not os.path.exists(f"aku/fine_tuning/dpo_data/record_{n}.json"):
            max_datafile_num = n
            break

    ds = Dataset.from_generator(gen, gen_kwargs={"max_datafile_num": max_datafile_num})

    print(f"Number of examples: {len(ds)}")

    ds = ds.train_test_split(test_size=0.05)
    print(ds)
    print("-"*50)
    print(ds["train"][0])

    login()
    ds.push_to_hub("YukiTomita-CC/AKU-d_ms-0.5B-chat-v0.1_dpo_dataset", private=True)
