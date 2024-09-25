import os
import time

from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


run_names = {
    "0": "baseline",
    "1": "curriculum",
    "2": "inc_batch",
    "3": "not_gqa",
    "4": "checkpoint",
    "5": "final_rlx8",
    }

def download_model(revision: str) -> None:
    snapshot_download(
        repo_id="YukiTomita-CC/comparison",
        revision=revision,
        local_dir=f"models/{revision}",
        ignore_patterns=["README.md", ".gitattributes"]
    )

def inference(model_name: str, device: str, manual_prompt: str = None) -> None:
    model_path = f"models/{model_name}"

    if not os.path.exists(model_path):
        print(f"Model {model_name} does not exist.")
        return
    
    print(f"Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print("Starting inference ...")
    if not manual_prompt:
        with open("data/test_set.txt", "r", encoding="utf-8") as f:
            test_set = f.readlines()

        while True:
            prompt = input("Enter a prompt: ")
            test_set = [prompt]
            
            for i, prompt in enumerate(test_set):
                prompt = prompt.strip().replace("__BR__", "\n")
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                input_ids = inputs["input_ids"]

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=128,
                        do_sample=True,
                        top_k=50,
                        top_p=0.9,
                        num_return_sequences=5,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # output = tokenizer.decode(tokens[0])

                for output_id in output_ids:
                    print(tokenizer.decode(output_id, skip_special_tokens=True))
                    print("\n" + "-"*50 + "\n")

                # print("-" * 10, f"Prompt_{str(i)}", "-" * 10)
                # print(f"{prompt} -> {output}")
                # print()
    
    else:
        pass


if __name__ == "__main__":
    device = "cuda"
    if not torch.cuda.is_available():
        device = "cpu"
        print("CUDA is not available. Continuing with CPU.")

    mode = input("Do you want to use a downloaded model or download a new checkpoint? (dl/new): ")
    print("Available run names: ", run_names)
    model_name, revision = input("""Please enter the model name(e.g. baseline_1epoch is "0 1epoch"): """).split()
    model_name = f"{run_names[model_name]}_{revision}"
    print("\n\n")

    if mode == "new":
        download_model(model_name)

    inference(model_name, device)
