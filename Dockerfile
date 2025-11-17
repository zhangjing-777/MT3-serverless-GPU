FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ---------------------------
# Install system dependencies
# ---------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    fluidsynth \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Install Python Dependencies
# ---------------------------
RUN pip install --upgrade pip && \
    pip install runpod \
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
# Download MT3 checkpoint
# ---------------------------
RUN mkdir -p /mt3/checkpoints && \
    cd /mt3/checkpoints && \
    gsutil -m cp -r gs://mt3/checkpoints/mt3/ .

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
