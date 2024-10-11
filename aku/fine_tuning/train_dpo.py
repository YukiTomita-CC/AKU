from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig


dataset = load_dataset("YukiTomita-CC/AKU-d_ms-0.5B-chat-v0.1_dpo_dataset", token="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")

train_dataset = dataset["train"]
val_dataset = dataset["test"]

model_name = "models/AKU-d_ms-0.5B-chat-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
)

training_args = DPOConfig(
    output_dir="aku/fine_tuning/output_dpo",
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    prediction_loss_only=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-6,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1.0e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    logging_dir="aku/fine_tuning/logs",
    logging_strategy="steps",
    logging_steps=2,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=3,
    save_safetensors=True,
    seed=42,
    fp16=True,
    eval_steps=1,
    dataloader_num_workers=4,
    run_name="inc_epoch",
    remove_unused_columns=True,
    optim="adamw_bnb_8bit",
    report_to="wandb",
)

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=0.3,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    max_prompt_length=187,
    max_length=512,
    loss_type="sigmoid",
    label_smoothing=0.1,
    precompute_ref_log_probs=True,
)

dpo_trainer.train()
