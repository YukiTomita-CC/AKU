FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    git

WORKDIR /workspace
