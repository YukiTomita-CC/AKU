import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class AkuModel:
    def __init__(self, model_path: str) -> None:
        device = "cuda"
        if not torch.cuda.is_available():
            device = "cpu"
            print("CUDA is not available. Continuing with CPU.")

        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    def _format_to_prompt_template(self, input_info: dict) -> str:
        prompt = f"<CONTEXT>{input_info['context']}</CONTEXT>\n"
        for dialog in input_info["dialogs"]:
            role = "User" if dialog["role"] == "user" else "Aku"
            prompt += f"\n<ROLE>{role}</ROLE>\n{dialog['content']}\n"
        
        prompt += f" <ATTR>likability: {input_info['likability']} mood: {input_info['mood']}</ATTR>\n<ROLE>Aku</ROLE>\n"

        return prompt

    def _inference(self, input_info: dict) -> None:
        prompt = self._format_to_prompt_template(input_info)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        with torch.no_grad():
            tokens = self.model.generate(
                **inputs,
                max_new_tokens=32
            )
        
        output = self.tokenizer.decode(tokens[0])

        attr_idx = output.rfind("</ATTR>")
        if attr_idx != -1:
            output = output[attr_idx + len("</ATTR>"):]
        
        output = output[len("<ROLE>Aku</ROLE> ") + 3:]

        role_idx = output.find("<ROLE>")
        if role_idx != -1:
            output = output[:role_idx].strip()

        return output

    def batch_inference(self, prompt: str, batch_num: int=5) -> list:
        return [self._inference(prompt) for _ in range(batch_num)]
