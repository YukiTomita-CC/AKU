import os
import time

from huggingface_hub import snapshot_download
import torch
from transformers import MistralForCausalLM, T5Tokenizer


run_names = {
    "0": "baseline",
    "1": "curriculum",
    "2": "inc_batch",
    "3": "not_gqa",
    "4": "checkpoint"
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
    model = MistralForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    print("Starting inference ...")
    if not manual_prompt:
        with open("data/test_set.txt", "r", encoding="utf-8") as f:
            test_set = f.readlines()
        
        for i, prompt in enumerate(test_set):
            prompt = prompt.strip().replace("__BR__", "\n")
            inputs = tokenizer(prompt, return_tensors="pt")
            
            if inputs['input_ids'][:, -1] == tokenizer.eos_token_id:
                inputs['input_ids'] = inputs['input_ids'][:, :-1]
                inputs['attention_mask'] = inputs['attention_mask'][:, :-1]
            inputs.to(model.device)
            # print(inputs)

            start_time = time.time()
            with torch.no_grad():
                tokens = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                )
            end_time = time.time()

            num_tokens = len(tokens[0]) - len(inputs['input_ids'][0])
            elapsed_time = end_time - start_time
            tps = num_tokens / elapsed_time
            
            output = tokenizer.decode(tokens[0], skip_special_tokens=True)
            # output = output.replace(' ', '\n') # might be unnecessary in the future

            print("-" * 10, f"Prompt_{str(i)}", "-" * 10)
            print(f"{prompt} -> {output}")
            print(f"[Time taken: {elapsed_time:.2f}s, Number of tokens: {num_tokens}, TPS: {tps:.2f}]")
            print()
    
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
