FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 via deadsnakes PPA
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential ninja-build && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    python3 -m ensurepip --upgrade && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip wheel setuptools

# PyTorch (pinned, must match pyproject.toml)
RUN python3 -m pip install torch==2.7.1 torchvision==0.22.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# CUDA extensions — the slow/fragile part that motivates this Dockerfile.
# Built here instead of in Modal because Modal's image builder has output rate
# limits and resource constraints that kill the verbose ptxas compilation.
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV MAX_JOBS="1"
RUN python3 -m pip install causal-conv1d --no-build-isolation
RUN python3 -m pip install mamba-ssm==2.2.5 --no-build-isolation

RUN python3 -m pip cache purge
