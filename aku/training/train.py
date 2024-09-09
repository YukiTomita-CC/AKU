from collections import OrderedDict
from itertools import chain
import json
import math
import os

from datasets import load_dataset
import evaluate
import torch
from transformers import (
    HfArgumentParser,
    T5Tokenizer,
    MistralConfig,
    MistralForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


def prepare_training():
    if os.path.basename(os.getcwd()) != "AKU":
        raise RuntimeError("This script must be executed in the 'AKU' directory. (project root)")

    prepare_dir = "aku/training/prepare"
    raw_data_dir = os.path.join(prepare_dir, "raw_data")
    processed_data_dir = os.path.join(prepare_dir, "processed_data")
    tokenizer_dir = os.path.join(prepare_dir, "tokenizer")
    
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    raw_files = [f for f in os.listdir(raw_data_dir) if os.path.isfile(os.path.join(raw_data_dir, f))]
    if not raw_files:
        raise RuntimeError("No raw data found. Please download it from HuggingFace.")

    required_processed_files = ["train.txt", "test.txt"]
    for file in required_processed_files:
        if not os.path.exists(os.path.join(processed_data_dir, file)):
            raise RuntimeError("Processed data not found. Please run 'create_corpus.sh'.")

    required_tokenizer_files = ["special_tokens_map.json", "spiece.model", "tokenizer_config.json"]
    for file in required_tokenizer_files:
        if not os.path.exists(os.path.join(tokenizer_dir, file)):
            raise RuntimeError("Tokenizer not found. Please download it from HuggingFace.")
    
    print("All checks passed. The environment is ready for training.")


def main():
    prepare_training()

    train_file = "aku/training/prepare/processed_data/train.txt"
    validation_file = "aku/training/prepare/processed_data/test.txt"
    tokenizer_name = "aku/training/prepare/tokenizer"

    print("[INFO] Parsing training arguments...")
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_json_file(json_file="aku/training/config/training_config.json")[0]

    set_seed(training_args.seed)


    # Dataset
    print("[INFO] Loading dataset...")
    data_files = {"train": train_file, "validation": validation_file}
    
    raw_datasets = load_dataset(
        "text",
        data_files=data_files,
    )

    print("[INFO] Preprocessing dataset... (Replacing '__BR__' with '\\n')")
    raw_datasets = raw_datasets.map(lambda examples: {"text": examples["text"].replace("__BR__", "\n")})

    print("[INFO] Tokenizing dataset...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

    column_names = list(raw_datasets["train"].features)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names
        )

    print("[INFO] Grouping dataset...")
    def group_texts(examples):
        block_size = 1024

        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # print(f"Train dataset length: {len(train_dataset)}")
    # print(f"Eval dataset length: {len(eval_dataset)}")

    # print("Train dataset example:")
    # for i in range(5):
    #     print(train_dataset["input_ids"][i])
    #     print(tokenizer.decode(train_dataset["input_ids"][i]))
    
    # print("Eval dataset example:")
    # for i in range(5):
    #     print(eval_dataset["input_ids"][i])
    #     print(tokenizer.decode(eval_dataset["input_ids"][i]))

    # input("Did the dataset look correct? Press Enter to continue...")

    # Model
    print("[INFO] Initializing model...")
    def load_config_from_json(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            config = MistralConfig.from_dict(config)
        return config

    config = load_config_from_json(config_file = "aku/training/config/mistral_0_5b_config.json")

    model = MistralForCausalLM.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=config, 
        state_dict=OrderedDict(),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
        )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    print("[INFO] Loading metric...")
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)


    # Initialize Trainer
    print("[INFO] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Check
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        raise ValueError(
            f"Your tokenizer has a vocab size of {len(tokenizer)}, but your model has a vocab size of {model.get_input_embeddings().weight.shape[0]}."
        )

    # Training
    print("[INFO] Strat training...")
    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    print("[INFO] Start evaluation...")
    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
