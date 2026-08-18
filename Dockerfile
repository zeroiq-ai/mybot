FROM python:3.12-slim

# ffmpeg for TTS/voice-note encoding and video frame sampling
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# HF Spaces storage is ephemeral on the free tier — keep the DB in /tmp.
ENV DB_PATH=/tmp/bot.db
# Hugging Face Spaces expect the container to listen on 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "space_app.py"]
