# Image chạy CameraAI trên mini PC / máy chủ cùng mạng LAN với IP camera (RTSP).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAMERAAI_HOST=0.0.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cameraai ./cameraai
COPY config.yaml .
# Tải model lúc build (kiểm SHA256) để container chạy được khi không có Internet.
RUN python -m cameraai.model_store download

EXPOSE 8000
CMD ["python", "-m", "cameraai"]
