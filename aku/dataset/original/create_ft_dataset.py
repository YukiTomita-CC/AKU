from datasets import Dataset
import json
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
        for i in range(len(data["conversation"])):
            if i % 2 == 1:
                input = []
                for j in range(i):
                    if j % 2 == 0:
                        input.append(replace_question_and_exclamation_marks(data["conversation"][j]))
                    else:
                        input.append(
                            {
                                "role": data["conversation"][j]["role"],
                                "content": replace_question_and_exclamation_marks(data["conversation"][j]["content"])
                            })

                yield {
                    "dialog_id": str(n),
                    "context": context,
                    "input": input,
                    "output": replace_question_and_exclamation_marks(data["conversation"][i]["content"]),
                    "likability": data["conversation"][i]["attribute"]["likability"],
                    "mood": data["conversation"][i]["attribute"]["mood"]
                }


if __name__ == "__main__":
    max_dialog_num = 10
    ds = Dataset.from_generator(gen, gen_kwargs={"max_dialog_num": max_dialog_num})

    login()
    ds.push_to_hub("YukiTomita-CC/AKU-d_ms-0.5B-chat-v0.1_dataset", private=True)
