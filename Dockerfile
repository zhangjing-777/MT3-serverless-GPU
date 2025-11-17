FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# 系统依赖
RUN apt-get update -qq && apt-get install -qq \
    libfluidsynth3 \
    build-essential \
    libasound2-dev \
    libjack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 用 pip 安装 gsutil
RUN pip install gsutil

# 直接安装 JAX CUDA 12（指定具体包）
RUN pip install jax jaxlib==0.4.20+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 安装 MT3
WORKDIR /content
RUN git clone --branch=main https://github.com/magenta/mt3 && \
    mv mt3 mt3_tmp && \
    mv mt3_tmp/* . && \
    rm -r mt3_tmp && \
    pip install nest-asyncio pyfluidsynth==1.3.0 runpod boto3 -e .

# 下载模型
RUN gsutil -q -m cp -r gs://mt3/checkpoints .

WORKDIR /app
COPY src/ ./src/

CMD ["python3", "-u", "src/handler.py"]
