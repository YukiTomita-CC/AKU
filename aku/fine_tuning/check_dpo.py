import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "aku/fine_tuning/output_dpo/checkpoint-98"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto").to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

messages = [
    {"role": "user", "content": "こんにちは!あなたの名前は何ですか?"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(input_ids, max_new_tokens=128, do_sample=True, top_p=0.9, top_k=50, num_return_sequences=5)

for output in outputs:
    print(tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True))
