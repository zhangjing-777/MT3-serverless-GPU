FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
# Force rebuild - v2
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
RUN pip install --upgrade pip && \
    pip install runpod \
    boto3 \
    jax[cuda12] \
    nest-asyncio \
    pyfluidsynth==1.3.0 \
    librosa \
    note_seq \
    t5[gcp] \
    t5x \
    seqio

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
