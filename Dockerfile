# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

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

# Expose port
EXPOSE 8080

# Run the Flask app
CMD exec python main.py

