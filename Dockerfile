FROM --platform=linux/amd64 python:3.11

WORKDIR /app

# Update package list, install ffmpeg, and clean up in one RUN command
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-cpu.txt .

RUN pip install --no-cache-dir -r requirements-cpu.txt

COPY . .

# Install gunicorn explicitly
RUN pip install gunicorn

# Create necessary directories
RUN mkdir -p checkpoints/auxiliary temp_api detect_results_api

# Explicitly set port and unset any Flask-specific port variables
ENV PORT=8080
ENV FLASK_RUN_PORT=8080

# Unset any other Flask environment variables that might override the port
ENV FLASK_APP=

# Make sure the container knows to listen on port 8080
EXPOSE 8080

CMD ["python", "main.py"]

