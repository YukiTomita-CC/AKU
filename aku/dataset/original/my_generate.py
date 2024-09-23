import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class UserResponseGenerator:
    def __init__(self, model_name: str) -> None:
        temp_model_name = "./model"
        self.model = AutoModelForCausalLM.from_pretrained(
            temp_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(temp_model_name)

    def generate_user_response(self, model_name, conversations):
        complexity = 0
        toxicity = 0
        humor = 0
        creativity = 0

        messages = conversations[:-1]
        messages.append({
            "role": "user",
            "content": conversations[-1]["content"],
            "complexity": complexity,
            "toxicity": toxicity,
            "humor": humor,
            "creativity": creativity,
        })

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=64,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=5
            )

        return [{"content": self.tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True)} for output in outputs]


if __name__ == "__main__":
    generator = UserResponseGenerator("")
    conversations = [
        { "role": "user", "content": "Hello, how are you?" },
        { "role": "assistant", "content": "I'm doing well, thank you." },
        { "role": "user", "content": "OK. Did you find anything funny?" },
    ]
    responses = generator.generate_user_response("", conversations)
    for i, response in enumerate(responses):
        print(f"Response {i + 1}: {response['content']}")
