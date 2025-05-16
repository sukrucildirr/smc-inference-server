# Use a NVIDIA CUDA base image. This is essential for GPU support.
# Choose a version that matches your CUDA toolkit and GPU driver compatibility.
# For example, nvcr.io/nvidia/pytorch:23.10-py3 or pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# I'll use a general CUDA runtime image for flexibility, assuming PyTorch/vLLM will handle specifics.
# If you encounter issues, try a more specific PyTorch/CUDA image.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=off \
    HF_HOME="/app/huggingface" \
    VLLM_HOME="/app/vllm-cache"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \ 
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy your requirements.txt first to leverage Docker cache
# Create a dummy requirements.txt if you don't have one, then add your actual deps.
COPY requirements.txt .

# Install Python dependencies
# Use --break-system-packages if encountering permission issues on system-wide python installs
# (less likely with python3-pip but good to know)
# Specifically install torch with CUDA support and vLLM from source for better compatibility.
# You might need to adjust the vLLM version or installation method based on your needs.
# For vLLM to work, you generally need to install it with specific CUDA support.
# Check vLLM's official documentation for the recommended installation for your CUDA version.
# Example for CUDA 12.1:
# pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# pip install vllm==0.3.3
#RUN pip install --upgrade pip setuptools wheel \
#    && pip install fastapi uvicorn==0.27.0 python-dotenv pydantic \
#    && pip install vllm \
#    && pip install numpy==2.1.3 \
#    && pip install llamppl
#COPY requirements.txt .

# Install them in the Docker image
RUN pip install --no-cache-dir -r requirements.txt

    #&& pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121 
    # && pip install vllm==0.3.3 \
    # && pip install llvmlite \
    # && pip install genlm-backend \
    # Install llamppl. If it's a local package or private repo, adjust.
    # Assuming it's pip-installable for now.
    # If llamppl is a local directory, you'll need to copy it and install:
    # COPY llamppl /app/llamppl
    # RUN pip install /app/llamppl
    # If it's on PyPI, it's just:
    # For now, let's assume it's part of your project and needs to be copied.
    # Let's adjust this assuming llamppl is a local directory in your project root.

# If llamppl is a local directory:
#COPY llamppl /app/llamppl/
#RUN pip install /app/llamppl

# Copy the rest of your application code
COPY . .

# Expose the port your FastAPI application will run on
EXPOSE 8000

# Command to run the application using Uvicorn.
# The `WORKER_ID` is set by the orchestration script later.
# We'll use 0.0.0.0 to make it accessible from outside the container.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]