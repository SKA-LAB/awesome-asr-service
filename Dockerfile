FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Copy code
WORKDIR /app
COPY . .

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint
CMD ["bash", "run.sh"]
