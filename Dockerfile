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

# 先装旧版本的 flax（兼容 jax 0.4.x）
RUN pip install flax==0.6.0 jax==0.4.20 jaxlib==0.4.20+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 安装 MT3（会跳过已安装的 jax/flax）
WORKDIR /content  
RUN git clone --branch=main https://github.com/magenta/mt3 && \
    mv mt3 mt3_tmp && \
    mv mt3_tmp/* . && \
    rm -r mt3_tmp && \
    pip install --no-deps nest-asyncio pyfluidsynth==1.3.0 runpod boto3 -e .

# 手动安装 MT3 的其他依赖（跳过 flax/jax）
RUN pip install note-seq t5 gin-config seqio-nightly tensorflow

# 下载模型
RUN gsutil -q -m cp -r gs://mt3/checkpoints .

WORKDIR /app
COPY src/ ./src/

CMD ["python3", "-u", "src/handler.py"]
