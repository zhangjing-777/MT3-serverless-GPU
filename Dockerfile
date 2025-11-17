FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ---------------------------
# Install system dependencies
# ---------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    fluidsynth \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Install Python Dependencies
# ---------------------------
RUN pip install --upgrade pip && \
    pip install runpod \
    boto3 \
    tensorflow==2.12.0 \
    note-seq \
    librosa \
    pretty-midi \
    mir-eval

# ---------------------------
# Clone MT3 repository
# ---------------------------
WORKDIR /mt3
RUN git clone https://github.com/magenta/mt3.git . && \
    pip install -e .

# ---------------------------
# Download MT3 checkpoint (手动下载，不用 gsutil)
# ---------------------------
RUN mkdir -p /mt3/checkpoints/mt3 && \
    cd /mt3/checkpoints && \
    wget -q https://storage.googleapis.com/magentadata/models/mt3/checkpoints/mt3.gin -O mt3/config.gin || echo "Config download failed" && \
    wget -q https://storage.googleapis.com/magentadata/models/mt3/checkpoints/checkpoint || echo "Checkpoint list failed"

# Note: MT3 模型可能需要在首次运行时下载

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
