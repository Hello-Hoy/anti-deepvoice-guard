FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드
COPY . /app/model_server/

# 모델 가중치 (빌드 시 복사하거나 볼륨 마운트)
# COPY models/weights/ /app/models/weights/

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/weights/AASIST.pth
ENV DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "model_server.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
