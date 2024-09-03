from transformers import (
    Trainer,
    TrainingArguments,
    MixtralForCausalLM,
    MixtralConfig,
    LlamaTokenizerFast,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, IterableDataset, DownloadConfig
from transformers import TrainingArguments
from argparse import ArgumentParser
from pathlib import Path


parser = ArgumentParser()
parser.add_argument("--resume", action="store_true")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--batch_steps", type=int, default=32)
args = parser.parse_args()


if __name__ == "__main__":
    ds: IterableDataset = load_dataset(
        "neody/kusanagi",
        split="train",
        download_config=DownloadConfig(resume_download=True),
        streaming=True,
    )
    ds = ds.select_columns("text")
    ds = ds.shuffle(1234)

    def filter_ds(col):
        return all(
            [
                col["text"] not in x
                for x in [
                    "ピル",
                    "ニキビ",
                    "法律",
                    "薬局",
                    "歯医者",
                    "健康",
                    "症例",
                    "避妊",
                    "取扱い",
                    "激安",
                    "小遣い",
                ]
            ]
        )

    ds = ds.filter(filter_ds, batched=False)
    tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(
        "./data/tokenizer"
    )

    context_length = 1024

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length <= context_length:
                input_batch.append(input_ids + [tokenizer.eos_token_id])
        return {"input_ids": input_batch}

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    block_size = context_length

    training_args = TrainingArguments(
        optim="adamw_8bit",
        output_dir="./data/model",
        logging_dir="./data/logs",
        logging_steps=5,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.batch_steps,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        max_steps=1_000_000,
        bf16=True,
        weight_decay=0.1,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        learning_rate=1e-4,
        max_grad_norm=1.5,
        report_to="tensorboard",
    )

    if not Path("./data/model-zero").exists():
        config = MixtralConfig(
            vocab_size=len(tokenizer),
            hidden_size=1024,
            num_hidden_layers=16,
            intermediate_size=1024,
            num_attention_heads=16,
        )
        model: MixtralForCausalLM = MixtralForCausalLM(config)
        model.save_pretrained("./data/model-zero")

    tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer),
    )
    trainer.train(args.resume)
