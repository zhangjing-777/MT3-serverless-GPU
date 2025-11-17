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
# Install Python Dependencies
# ---------------------------
RUN pip install --upgrade pip

# 基础依赖
RUN pip install runpod boto3 librosa nest-asyncio pyfluidsynth==1.3.0

# JAX
RUN pip install "jax[cuda12]==0.4.20" jaxlib==0.4.20+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 其他依赖
RUN pip install note-seq

# ---------------------------
# Clone and install MT3 (会自动安装 t5, t5x, seqio)
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
