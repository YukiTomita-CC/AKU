from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


dataset = load_dataset("YukiTomita-CC/AKU-d_ms-0.5B-chat-v0.1_dataset", token="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")
print(f"Dataset size: {len(dataset)}")

model_path = "models/final_rlx8_5epoch"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

def convert_role(role):
    if role == "user":
        return "User"
    elif role == "assistant":
        return "Aku"
    else:
        raise ValueError("Invalid role.")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['dialog_id'])):
        text = ""
        for j in range(len(example['input'][i])):
            if j == 0:
                text += f"<ROLE>{convert_role(example['input'][i][j]['role'])}</ROLE>\n{example['input'][i][j]['content']}"
            else:
                text += f"\n<ROLE>{convert_role(example['input'][i][j]['role'])}</ROLE>\n{example['input'][i][j]['content']}"
        
        text += f" <ATTR> likability: {example['likability'][i]} mood: {example['mood'][i]} </ATTR>"
        text += f"\n<ROLE>Aku</ROLE>\n{example['output'][i]}<EOS>"

        output_texts.append(text)
    return output_texts

response_template = [6, 9, 7]
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False
)

def model_init(trial=None):
    torch.cuda.empty_cache()
    return AutoModelForCausalLM.from_pretrained(model_path)

def wandb_hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "eval_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"values": [1e-6, 4e-6, 8e-6, 1e-5, 4e-5, 8e-5, 1e-4, 4e-4, 8e-4]},
            "num_train_epochs": {"values": [1, 2, 3, 4, 5]},
            "lr_scheduler_type": {"values": ["constant", "linear", "cosine"]},
            "neftune_noise_alpha": {"values": [5, 10, 15]},
        },
    }


def get_config(learning_rate, num_train_epochs, lr_scheduler_type, neftune_noise_alpha):
    return SFTConfig(
        output_dir="aku/fine_tuning/output",
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        prediction_loss_only=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1.0e-4,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=lr_scheduler_type,
        logging_dir="aku/fine_tuning/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        save_safetensors=True,
        seed=42,
        fp16=True,
        eval_steps=20,
        dataloader_num_workers=4,
        run_name="sweep_run",
        remove_unused_columns=True,
        optim="adamw_bnb_8bit",
        report_to="wandb",
        neftune_noise_alpha=neftune_noise_alpha,
        packing=False,
    )

trainer = SFTTrainer(
    model_init=model_init,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=get_config(1e-6, 1, "constant", 5),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="wandb",
    hp_space=wandb_hp_space,
    n_trials=10,
)

print(f"Best trial results: {best_trial}")
