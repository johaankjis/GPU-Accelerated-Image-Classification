# GPU-Accelerated Image Classification Deployment
# Optimized for NVIDIA GPUs with CUDA support

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
RUN pip3 install \
    numpy \
    pillow \
    matplotlib \
    onnx \
    onnxruntime-gpu

# Copy application files
COPY scripts/ /app/scripts/
COPY data/ /app/data/ 2>/dev/null || true
COPY *.pth /app/ 2>/dev/null || true

# Set environment variables for reproducibility
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create data directory
RUN mkdir -p /app/data

# Default command runs inference
CMD ["python3", "scripts/inference.py"]
