#!/bin/bash
source .venv/bin/activate

.venv/bin/python -m pip install --upgrade pip

.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

.venv/bin/pip install -r requirements/requirements_train.txt
.venv/bin/pip install flash-attn==2.3.4 --no-build-isolation

.venv/bin/pip install wandb
wandb login
export WANDB_PROJECT=AKU-d_ms-0_5B-v0_1

huggingface-cli download YukiTomita-CC/AKU-d_ms-0.5B-v0.1_dataset --include "*.txt" --repo-type dataset --local-dir aku/training/dataset/raw
huggingface-cli download YukiTomita-CC/AKU-d_ms_tokenizer --local-dir aku/training/tokenizer/custom

bash aku/training/create_corpus.sh
