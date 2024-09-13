from datasets import load_dataset
from huggingface_hub import login
from transformers import MistralForCausalLM, T5Tokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


login()

dataset = load_dataset("YukiTomita-CC/AKU-d_ms-0.5B-chat-v0.1_dataset", split="train")

model_path = "models/inc_batch_2epoch"
model = MistralForCausalLM.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

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
        text = f"<CONTEXT>{example['context'][i]}</CONTEXT>\n"
        for j in range(len(example['input'][i])):
            text += f"\n<ROLE>{convert_role(example['input'][i][j]['role'])}</ROLE>\n{example['input'][i][j]['content']}"
        
        text += f" <ATTR> likability: {example['likability'][i]} mood: {example['mood'][i]} </ATTR>"
        text += "\n<ROLE>Aku</ROLE>\n"

        output_texts.append(text)
    return output_texts

response_template = "<ROLE>Aku</ROLE>\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False
    )

config = SFTConfig(
    output_dir="aku/fine_tuning/output",
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    prediction_loss_only=False, # ?
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_dir="aku/fine_tuning/logs",
    logging_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    save_safetensors=True,
    seed=42,
    fp16=True,
    eval_steps=100,
    dataloader_num_workers=4,
    run_name="test",
    remove_unused_columns=True,
    optim="adamw_bnb_8bit",
    report_to="wandb",
    neftune_noise_alpha=0.1,
    packing=False,
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# trainer.train()
