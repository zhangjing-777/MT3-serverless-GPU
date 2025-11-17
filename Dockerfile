FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# 系统依赖
RUN apt-get update -qq && apt-get install -qq \
    libfluidsynth3 \
    build-essential \
    libasound2-dev \
    libjack-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 用 pip 安装 gsutil
RUN pip install gsutil

# 安装 MT3
WORKDIR /content
RUN git clone --branch=main https://github.com/magenta/mt3 && \
    mv mt3 mt3_tmp && \
    mv mt3_tmp/* . && \
    rm -r mt3_tmp && \
    python3 -m pip install jax[cuda12] nest-asyncio pyfluidsynth==1.3.0 runpod boto3 -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 下载模型
RUN gsutil -q -m cp -r gs://mt3/checkpoints .

WORKDIR /app
COPY src/ ./src/

CMD ["python3", "-u", "src/handler.py"]
