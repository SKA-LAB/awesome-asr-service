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
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')"

# Create directory for temporary files
RUN mkdir -p /app/tmp
RUN mkdir /meeting-notes
RUN mkdir /meeting-recordings

# Copy application code
COPY app/ /app/app/
COPY run.sh /app/
COPY chunk_mp3.sh /app/
COPY .env .

# Make run script executable
RUN chmod +x /app/run.sh
RUN chmod +x /app/chunk_mp3.sh

# Set environment variables
ENV TEMP_DIR=/app/tmp
ENV WHISPER_MODEL_SIZE=base
ENV PORT=8000
ENV DASHBOARD_PORT_1=8501
ENV DASHBOARD_PORT_2=8502
ENV WORKERS=1
ENV PYTHONPATH=/app
ENV REDIS_URL=redis://redis:6379/0
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OUTPUT_DIR=/meeting-notes
ENV BATCH_DIRECTORY=/meeting-recordings
ENV CHUNK_SCRIPT=/app/chunk_mp3.sh

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501
EXPOSE 8502

# Run the application with the script
CMD ["/app/run.sh"]