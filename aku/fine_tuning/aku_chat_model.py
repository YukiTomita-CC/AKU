import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class AkuChatModel:
    def __init__(self):
        self.model_name = "models/AKU-d_ms-0.5B-chat-v0.2"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float32, device_map="auto").to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
    
    def generate_responses(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model.generate(input_ids, max_new_tokens=128, do_sample=True, top_p=0.9, top_k=50, num_return_sequences=10)
        
        return [self.tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True) for output in outputs]
