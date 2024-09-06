#!/bin/bash

cd workspace || exit

export HOME=/workspace
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH

apt update && apt install -y \
    git \
    git-lfs \
    curl \
    wget \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libbz2-dev \
    libnss3-dev \
    libsqlite3-dev \
    libssl-dev \
    liblzma-dev \
    libreadline-dev \
    libffi-dev \
    libgl1-mesa-dev \
    locales \
    fish \
    vim \
    nano \
    iputils-ping \
    net-tools \
    software-properties-common \
    fonts-powerline


git clone https://github.com/pyenv/pyenv.git ~/.pyenv

echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.11.6
pyenv global 3.11.6

git clone https://github.com/YukiTomita-CC/AKU.git
cd AKU || exit

python -m venv --prompt . .venv
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install -r requirements/requirements_train.txt
pip install flash-attn==2.3.4 --no-build-isolation

pip install wandb
wandb login
export WANDB_PROJECT=AKU-d_ms-0_5B-v0_1

huggingface-cli download YukiTomita-CC/AKU-d_ms-0.5B-v0.1_dataset --include "*.txt" --exclude "wikipedia_ja.txt" --repo-type dataset --local-dir aku/training/prepare/raw_data
huggingface-cli download YukiTomita-CC/AKU-d_ms_tokenizer --local-dir aku/training/prepare/tokenizer
