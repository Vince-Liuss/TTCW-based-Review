FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH="/root/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64" \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONNOUSERSITE=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

# Python 3.12 is default on Ubuntu 24.04 — no PPA needed
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        python3.12 python3-pip && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && \
    rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# PyTorch 2.9.0 (cu128) — must come before flash-attn
RUN uv pip install --system --no-cache-dir \
        torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# flash-attn — needs torch + packaging present, disable build isolation
RUN uv pip install --system --no-cache-dir packaging && \
    uv pip install --system --no-cache-dir "flash-attn>=2.7.0" --no-build-isolation

# vLLM
RUN uv pip install --system --no-cache-dir vllm==0.12.0 --torch-backend=cu128 && \
    python -c "import vllm; print('vllm', vllm.__version__)"

# Project dependencies
RUN uv pip install --system --no-cache-dir \
        "accelerate>=1.7.0" \
        "bitsandbytes>=0.46.0" \
        "datasets>=3.6.0" \
        "deepspeed>=0.17.0" \
        "evaluate" \
        "bert-score" \
        "huggingface-hub>=0.32.4" \
        "jsonlines>=4.0.0" \
        "liger-kernel>=0.5.10" \
        "matplotlib>=3.10.3" \
        "peft>=0.15.2" \
        "safetensors>=0.5.3" \
        "scikit-learn>=1.7.0" \
        "scipy>=1.15.3" \
        "seaborn>=0.13.2" \
        "sentence-transformers>=4.1.0" \
        "spacy>=3.8.7" \
        "tokenizers>=0.21.1" \
        "transformers>=5.0.0" \
        "trl>=0.19.0" \
        "wandb>=0.20.1" \
        "torchao>=0.16.0" && \
    python -m spacy download en_core_web_md

WORKDIR /workspace
CMD ["python"]
