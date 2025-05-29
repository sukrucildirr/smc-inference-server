FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=off \
    HF_HOME="/app/huggingface" \
    VLLM_HOME="/app/vllm-cache"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \ 
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/server.py /app/server.py
COPY src/util /app/util

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]