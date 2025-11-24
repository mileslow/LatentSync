# CPU-Only Sync Detection API Docker Image
# 
# This Dockerfile builds a lightweight CPU-only container for the audio-visual
# sync detection API. It excludes GPU/CUDA dependencies and video generation
# packages to minimize image size.
#
# Build: docker build --platform linux/amd64 -t sync-detect .
# Run: docker run -p 8080:8080 sync-detect
#
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (CPU-only, no CUDA)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements for Docker build
# Note: Using requirements_docker.txt instead of requirements.txt to exclude
# heavy dependencies only needed for video generation (diffusers, transformers, etc.)
COPY requirements_docker.txt .

# Install PyTorch CPU-only version first (explicitly)
# This prevents pip from installing the larger CUDA-enabled version
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies (minimal set for sync detection API)
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints/auxiliary temp_api detect_results_api

# Download required model files
# SyncNet model (add download logic if needed)
RUN if [ ! -f "checkpoints/auxiliary/syncnet_v2.model" ]; then \
    echo "Warning: syncnet_v2.model not found. Please ensure it's included in the build."; \
    fi

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
# Force CPU-only operation
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_DEVICE=cpu

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Run the Flask app
CMD exec python main.py

