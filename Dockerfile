FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ---------------------------
# Install system dependencies
# ---------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    fluidsynth \
    build-essential \
    libasound2-dev \
    libjack-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Install gsutil
# ---------------------------
RUN curl https://sdk.cloud.google.com | bash
ENV PATH=$PATH:/root/google-cloud-sdk/bin

# ---------------------------
# Install Python Dependencies (分步安装避免冲突)
# ---------------------------
RUN pip install --upgrade pip

# 基础依赖
RUN pip install runpod boto3 librosa nest-asyncio pyfluidsynth==1.3.0

# JAX (单独安装)
RUN pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# T5 和 seqio
RUN pip install note-seq seqio

# T5X 相关
RUN pip install "t5[gcp]" t5x

# ---------------------------
# Clone and install MT3
# ---------------------------
WORKDIR /content
RUN git clone --branch=main https://github.com/magenta/mt3 && \
    cd mt3 && \
    pip install -e .

# ---------------------------
# Download MT3 checkpoints
# ---------------------------
RUN gsutil -q -m cp -r gs://mt3/checkpoints /content/checkpoints

# ---------------------------
# Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# Copy source code
# ---------------------------
COPY src/ ./src/

# ---------------------------
# Serverless entrypoint
# ---------------------------
CMD ["python3", "-u", "src/handler.py"]
