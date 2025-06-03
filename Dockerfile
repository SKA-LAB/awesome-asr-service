FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Update this line before the pip install command
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download the ASR model during build time
RUN python -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v2')"

# Create directory for temporary files
RUN mkdir -p /app/tmp

# Copy application code
COPY app/ /app/app/
COPY run.sh /app/

# Make run script executable
RUN chmod +x /app/run.sh

# Set environment variables
ENV TEMP_DIR=/app/tmp
ENV PORT=8000
ENV DASHBOARD_PORT=8501
ENV WORKERS=1
ENV PYTHONPATH=/app
ENV REDIS_URL=redis://redis:6379/0
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Run the application with the script
CMD ["/app/run.sh"]