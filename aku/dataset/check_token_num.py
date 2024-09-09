import os

from tqdm import tqdm
from transformers import T5Tokenizer


def count_tokens_in_file(file_path, tokenizer):
    total_tokens = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            tokens = tokenizer.encode(line.strip())
            token_count = len(tokens)
            total_tokens += token_count
    return total_tokens

def process_files(input_path, model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    files_to_process = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.txt')]
    
    grand_total_tokens = 0
    for file_path in files_to_process:
        file_total_tokens = count_tokens_in_file(file_path, tokenizer)
        grand_total_tokens += file_total_tokens
        print(f"{file_path}: {file_total_tokens} tokens")
    
    print(f"Total tokens across all files: {grand_total_tokens}")


if __name__ == "__main__":
    process_files("aku/training/prepare/raw_data", "aku/training/prepare/tokenizer")
