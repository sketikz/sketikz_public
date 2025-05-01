FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables to reduce interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-latex-recommended \
    latexmk \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the files
WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV PATH="/root/.local/bin:$PATH"
ENV STREAMLIT_WATCHER_TYPE none

RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.6/flash_attn-2.6.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl

COPY . .

# Default command
ENTRYPOINT ["streamlit", "run", "app.py"]
