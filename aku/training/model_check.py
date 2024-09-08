from transformers import MistralForCausalLM, MistralConfig
import json


def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = MistralConfig.from_dict(config)
    return config

config = load_config_from_json(config_file = "aku/training/config/mistral_0_5b_config.json")
print(config)

model = MistralForCausalLM(config)
print(model)

model_size = sum(t.numel() for t in model.parameters())
print(f"Size: {model_size/1000**2:.0f}M parameters")
print()

for name, param in model.named_parameters():
    print(f"Layer: {name:<47} | Shape: {param.shape}")
