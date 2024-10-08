import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class AkuModel:
    def __init__(self, model_path: str) -> None:
        device = "cuda"
        if not torch.cuda.is_available():
            device = "cpu"
            print("CUDA is not available. Continuing with CPU.")

        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    def inference(self, context: str, prompt: str, likability: str, mood: str) -> None:
        prompt = f"<CONTEXT>{context}</CONTEXT>\n\n<ROLE>User</ROLE>\n{prompt} <ATTR> likability: {likability} mood: {mood} </ATTR>\n<ROLE>Aku</ROLE>\n"
        prompt = prompt.strip().replace("？", "?").replace("！", "!")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
            ).to(self.model.device)

        start_time = time.time()
        with torch.no_grad():
            tokens = self.model.generate(
                **inputs,
                max_new_tokens=64
            )
        end_time = time.time()

        num_tokens = len(tokens[0]) - len(inputs['input_ids'][0])
        elapsed_time = end_time - start_time
        tps = num_tokens / elapsed_time
        
        output = self.tokenizer.decode(tokens[0])

        print("-" * 10, "Prompt", "-" * 10)
        print(f"{prompt} -> {output}")
        print(f"[Time taken: {elapsed_time:.2f}s, Number of tokens: {num_tokens}, TPS: {tps:.2f}]")
        print()


if __name__ == "__main__":
    model_path = "models/final_5epoch_ft1150"
    if not os.path.exists(model_path):
        print(f"Model {model_path} does not exist.")
        exit(1)
    
    print(f"Loading model ...")
    model = AkuModel(model_path)

    print("Starting inference ...")
    while True:
        context = input("Enter a context: ")
        if context == "":
            context = "*UserとAkuは久々に会って二人で話している"

        prompt = input("Enter a prompt: ")

        if prompt == "exit":
            break
    
        likability = input("Enter likability: ")
        mood = input("Enter mood: ")

        if likability == "":
            likability = "3"
        if mood == "":
            mood = "3"

        model.inference(context, prompt, likability, mood)
