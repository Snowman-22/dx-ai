FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 이미지 안에 `recommendation_algorithm/` 가 포함될 때 파이프라인이 동작하도록 기본값.
# EC2에서 /data 에 마운트만 쓰는 경우 `docker run -e RECOMMENDATION_ALGORITHM_PATH=/data/recommendation_algorithm` 로 덮어쓰면 됨.
ENV RECOMMENDATION_ALGORITHM_PATH=/app/recommendation_algorithm

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]