import json
import os

from datasets import Dataset
from datasets import load_dataset, load_from_disk
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

        try:
            num_conversations = len(data["conversations"])
            
            if num_conversations <= 10:
                input = [
                    replace_question_and_exclamation_marks(conversation)
                    if i % 2 == 0 else
                    {
                        "role": conversation["role"],
                        "content": replace_question_and_exclamation_marks(conversation["content"])
                    }
                    for i, conversation in enumerate(data["conversations"])
                ]

                yield {
                    "dialog_id": str(n),
                    "input": input
                }
            else:
                for i in range(0, num_conversations, 10):
                    input = [
                        replace_question_and_exclamation_marks(data["conversations"][j])
                        if j % 2 == 0 else
                        {
                            "role": data["conversations"][j]["role"],
                            "content": replace_question_and_exclamation_marks(data["conversations"][j]["content"])
                        }
                        for j in range(i, min(i + 10, num_conversations))
                    ]

                    yield {
                        "dialog_id": f"{n}_{i // 10}",
                        "input": input
                    }

        except Exception as e:
            print(f"Error in conversation {n}: {e}")


if __name__ == "__main__":
    # max_dialog_num = 0
    # for n in range(2000):
    #     if not os.path.exists(f"aku/dataset/original/conversations/conv_{n}.json"):
    #         max_dialog_num = n
    #         break

    # ds = Dataset.from_generator(gen, gen_kwargs={"max_dialog_num": max_dialog_num})

    # print(f"Number of examples: {len(ds)}")

    # ds = ds.train_test_split(test_size=0.05)
    # print(ds)

    # ds.save_to_disk("aku/dataset/original/aku-d_ms-0.5B-chat-v0.1_dataset")

    # login()
    # ds.push_to_hub("YukiTomita-CC/AKU-d_ms-0.5B-chat-v0.1_dataset", private=True)

    ds = load_from_disk(r"aku\dataset\original\aku-d_ms-0.5B-chat-v0.1_dataset")
    print(ds["train"][1])
