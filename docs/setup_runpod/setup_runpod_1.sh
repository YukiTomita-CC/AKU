#!/bin/bash
cd workspace
export HOME=/workspace

apt-get update && apt-get install -y git git-lfs curl wget build-essential \
    zlib1g-dev libncurses5-dev libgdbm-dev libbz2-dev libnss3-dev libsqlite3-dev \
    libssl-dev liblzma-dev libreadline-dev libffi-dev libgl1-mesa-dev locales \
    fish vim nano iputils-ping net-tools software-properties-common fonts-powerline screen

apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 130
update-alternatives --set python /usr/bin/python3.11

python --version

git clone https://github.com/YukiTomita-CC/AKU.git
cd AKU

mkdir -p log

python -m venv --prompt . .venv
