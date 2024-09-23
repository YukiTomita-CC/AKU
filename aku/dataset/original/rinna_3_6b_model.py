import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "models/rinna--japanese-gpt-neox-3.6b-instruction-ppo"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype="auto").to("cuda")

prompt = [
    {
        "speaker": "システム",
        "text": "やっほー、最近何か面白いことあった？"
    },
    {
        "speaker": "ユーザー",
        "text": "えー、面白いこと？🙄最近新しい漫画読み始めたことくらいかな？"
    }
]
prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in prompt
]
prompt = "<NL>".join(prompt)
prompt = (
    prompt
    + "<NL>"
    + "システム: "
)
print(prompt.replace("<NL>", "\n"), end="")

token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")

with torch.no_grad():
    output_ids = model.generate(
        token_ids,
        do_sample=True,
        max_new_tokens=32,
        temperature=0.95,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
# output = output.replace("<NL>", "\n")
print(output)
