from collections import OrderedDict
from itertools import chain
import json
import math

import evaluate
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    MistralConfig,
    MistralForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


train_file = "data/train/train.txt"
validation_file = "data/train/test.txt"
tokenizer_name = "data/train/tokenizer"

def main():
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_json_file(json_file="training_config.json")

    set_seed(training_args.seed)

    # Dataset
    data_files = {"train": train_file, "validation": validation_file}

    raw_datasets = load_dataset(
        "text",
        data_files=data_files,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    column_names = list(raw_datasets["train"].features)

    def tokenize_function(examples):
        replace_text = examples["text"].replace("__BR__", "\n")

        return tokenizer(replace_text)

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names
        )

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


    # Model
    def load_config_from_json(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            config = MistralConfig.from_dict(config)
        return config

    config = load_config_from_json(config_file = "mistral_0_5b_config.json")

    model = MistralForCausalLM.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=config, 
        state_dict=OrderedDict(),
        use_flash_attention_2=True
        )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    # Initialize Trainer
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

    # Training
    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
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
