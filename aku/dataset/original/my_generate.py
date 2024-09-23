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

        prompt = """あなたのタスクは「日常会話コーパスの作成を補助すること」です。
以下の設定に基づいて、自然で一貫性のある会話を生成してください：

あなたのペルソナ:
- 社会人の男性
- 趣味：食べること、映画鑑賞
- 話し方：フレンドリーで穏やか

状況設定：
- パートナーであるAkuと家で二人でのんびりしている

会話のトピック：
- 今日のご飯について

会話のスタイル：
- カジュアルで自然な会話
- 1回の発言は1〜3文程度で簡潔に

指示：
会話のトピックについて話を展開してください。質問や提案、相槌を交えながら、自然な会話の流れを作ってください。

では、コーパスの作成を開始します。上記の設定に基づいて、会話の最初の発言を生成してください。"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        for conv in conversations[:-1]:
            if conv["role"] == "user":
                messages.append({
                    "role": "assistant",
                    "content": conv["content"]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": conv["content"]
                })

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
