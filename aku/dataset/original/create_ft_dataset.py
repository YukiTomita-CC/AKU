import json
import os

from datasets import Dataset
from huggingface_hub import login


def replace_question_and_exclamation_marks(example):
    if isinstance(example, str):
        return example.replace("？", "?").replace("！", "!")

    elif isinstance(example, dict):
        return {
            "role": example["role"],
            "content": example["content"].replace("？", "?").replace("！", "!")
        }

    else:
        raise ValueError("Invalid type.")

def gen(max_dialog_num: int):
    for n in range(max_dialog_num):
        with open(f"aku/dataset/original/conversations/conv_{n}.json", "r", encoding="utf-8") as f:
            data = f.read()

        data = json.loads(data)
        context = data["context"]
        print(f"Length of data: {len(data['conversations'])}")
        for i in range(len(data["conversations"])):
            if i % 2 == 1:
                input = []
                for j in range(i):
                    if j % 2 == 0:
                        input.append(replace_question_and_exclamation_marks(data["conversations"][j]))
                    else:
                        input.append(
                            {
                                "role": data["conversations"][j]["role"],
                                "content": replace_question_and_exclamation_marks(data["conversations"][j]["content"])
                            })
                
                if len(input) > 9:
                    input = input[-9:]

                yield {
                    "dialog_id": str(n),
                    "context": context,
                    "input": input,
                    "output": replace_question_and_exclamation_marks(data["conversations"][i]["content"]),
                    "likability": data["conversations"][i]["attribute"]["likability"],
                    "mood": data["conversations"][i]["attribute"]["mood"]
                }


if __name__ == "__main__":
    max_dialog_num = 0
    for n in range(1000):
        if not os.path.exists(f"aku/dataset/original/conversations/conv_{n}.json"):
            max_dialog_num = n
            break

    ds = Dataset.from_generator(gen, gen_kwargs={"max_dialog_num": max_dialog_num})

    print(f"Number of examples: {len(ds)}")

    ds = ds.train_test_split(test_size=0.05)
    print(ds)

    login()
    ds.push_to_hub("YukiTomita-CC/AKU-d_ms-0.5B-chat-v0.1_dataset", private=True)
