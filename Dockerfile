FROM python:3.10-slim

# System deps: ffmpeg + OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU (install before other deps to avoid pip resolving the GPU wheel)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-create directories
RUN mkdir -p uploads Wav2Lip/temp Wav2Lip/checkpoints

EXPOSE 5001

CMD ["python", "server.py"]
