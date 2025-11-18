FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ---------------------------
# Install system dependencies (分开安装，更稳定)
# ---------------------------
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        fluidsynth \
        build-essential \
        libasound2-dev \
        libjack-dev \
        curl \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------
# Install gsutil
# ---------------------------
RUN curl https://sdk.cloud.google.com | bash
ENV PATH=$PATH:/root/google-cloud-sdk/bin

# ---------------------------
# Fix NumPy version first
# ---------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0"

# ---------------------------
# Install Python Dependencies
# ---------------------------
RUN pip install --no-cache-dir \
    runpod \
    boto3 \
    nest-asyncio \
    pyfluidsynth==1.3.0 \
    librosa \
    note_seq \
    t5[gcp] \
    t5x \
    seqio

RUN pip install --no-cache-dir "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ---------------------------
# Clone and install MT3
# ---------------------------
WORKDIR /content
RUN git clone --branch=main https://github.com/magenta/mt3 && \
    cd mt3 && \
    pip install --no-cache-dir -e .

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
