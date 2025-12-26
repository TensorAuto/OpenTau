FROM ubuntu:22.04

# Disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# cmake and build-essential for building some python packages
# ffmpeg for video processing
# git for installing dependencies from git
# libgl1 and libglib2.0-0 for opencv and potential gui deps
# python3.10 and related tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /lerobot

# Copy project files
COPY . .

# Create virtual environment and install dependencies
# We explicitly use the system python 3.10 for the venv
RUN uv venv .venv --python /usr/bin/python3.10 && \
    . .venv/bin/activate && \
    uv sync --extra tau0 --extra test --extra video_benchmark \
    --extra accelerate --extra dev --extra feetech \
    --extra openai --extra onnx --extra smolvla \
    --extra libero --extra metaworld

# Set environment variables
ENV PATH="/lerobot/.venv/bin:$PATH"
ENV LEROBOT_HOME="/lerobot"

# Default command
CMD ["/bin/bash"]
